import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_albert import AlbertPreTrainedModel, AlbertLayerNorm, AlbertLayerGroup
from .modeling_bert import BertEmbeddings
from .modeling_highway_bert import BertPooler

import numpy as np

def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    exp_x = torch.exp(x)
    A = torch.sum(exp_x, dim=1)    # sum of exp(x_i)
    B = torch.sum(x*exp_x, dim=1)  # sum of x_i * exp(x_i)
    return torch.log(A) - B/A

class AlbertEmbeddings(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        #super(AlbertEmbeddings, self).__init__()
        super().__init__(config)
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        #self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        #self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm = AlbertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

    #def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
    #     if input_ids is not None:
    #         input_shape = input_ids.size()
    #     else:
    #         input_shape = inputs_embeds.size()[:-1]
    #
    #     seq_length = input_shape[1]
    #     device = input_ids.device if input_ids is not None else inputs_embeds.device
    #     if position_ids is None:
    #         position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    #         position_ids = position_ids.unsqueeze(0).expand(input_shape)
    #     if token_type_ids is None:
    #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    #
    #     if inputs_embeds is None:
    #         inputs_embeds = self.word_embeddings(input_ids)
    #     position_embeddings = self.position_embeddings(position_ids)
    #     token_type_embeddings = self.token_type_embeddings(token_type_ids)
    #
    #     embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    #     embeddings = self.LayerNorm(embeddings)
    #     #embeddings = self.dropout(embeddings)
    #     return embeddings

class AlbertTransformer(nn.Module):
    def __init__(self, config, params):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config, params) for _ in range(config.num_hidden_groups)])

        self.entropy_predictor = config.entropy_predictor
        if config.entropy_predictor:
            self.lookup_table = np.loadtxt(config.lookup_table_file, delimiter=",")
            self.predict_layer = config.predict_layer
            self.predict_average_layers = config.predict_average_layers
            self.extra_layer=config.extra_layer
            self.get_predict_acc=config.get_predict_acc
            self.no_ee_before=config.no_ee_before

        #self.layer = nn.ModuleList([AlbertLayer(config) for _ in range(config.num_hidden_layers)])
        ### try grouping for efficiency
        if config.one_class:
            self.highway = nn.ModuleList([AlbertHighway(config) for _ in range(config.num_hidden_groups)])
            self.early_exit_entropy = [-1 for _ in range(config.num_hidden_groups)]
        else:
            self.highway = nn.ModuleList([AlbertHighway(config) for _ in range(config.num_hidden_layers)])
            self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]


    def set_early_exit_entropy(self, x):
        print(x)
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()
        all_highway_exits = ()
        #if self.output_hidden_states:
        #    all_hidden_states = (hidden_states,)

        #for i,layer_module in enumerate(self.albert_layer_groups):
        #for i, layer_module in enumerate(self.layer):
        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

           #stopped here
            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            #added this section
            current_outputs = (hidden_states,)
            if self.output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if self.output_attentions:
                current_outputs = current_outputs + (all_attentions,)

            if self.config.one_class:
                highway_exit = self.highway[group_idx](current_outputs)
            else:
                highway_exit = self.highway[i](current_outputs)

            #added this section
            if not self.training:
                highway_logits = highway_exit[0]
                highway_entropy = entropy(highway_logits)
                highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                all_highway_exits = all_highway_exits + (highway_exit,)

                if self.config.one_class:
                    ent_ = self.early_exit_entropy[group_idx]
                else:
                    ent_ = self.early_exit_entropy[i]

                if not self.entropy_predictor:
                    if highway_entropy < ent_:
                        new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                        raise HighwayException(new_output, i+1)

                elif (self.get_predict_acc):
                    if i==0:
                        count = 0
                        check_ee = 0
                    if self.predict_layer-1 == i:
                        if self.predict_average_layers:
                            if i == 0:
                                hw_ent_temp = highway_entropy.cpu().numpy()[0]
                            else:
                                hw_ent_temp = hw_ent_temp + highway_entropy.cpu().numpy()[0]
                            hw_ent = hw_ent_temp / float((i+1))
                        else:
                            hw_ent = highway_entropy.cpu().numpy()[0]
                        #hash into lookup table w/ highway_entropy
                        idx = (np.abs(self.lookup_table[:,0] - hw_ent)).argmin()
                        entropy_layers = np.transpose(self.lookup_table[idx,1:])
                        below_thresh = entropy_layers < ent_
                        k = np.argmax(below_thresh) # k is number of remaining layers
                        if (np.sum(below_thresh) == 0): #never hit threshold
                            k = entropy_layers.shape[0] - 1
                        k = k + self.predict_layer
                        count = count + 1
                        #print(idx)
                        #print(self.lookup_table[idx,:])
                        #print(k)

                    if ((highway_entropy < ent_) or (i == self.config.num_hidden_layers-1)) and not check_ee:
                        j = i # j is hw exit layer
                        count = count + 1
                        check_ee = 1

                    if count == 2:
                        new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                        #return abs value of diff between j and k
                        if j>k:
                          raise HighwayException(new_output, (j-k) + 1)
                        else:
                          raise HighwayException(new_output, (k-j) + 1)

                else:
                    if (i < self.predict_layer - 1): # before predict layer
                        #exit here????
                        if highway_entropy < ent_:
                            new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                            raise HighwayException(new_output, i+1)

                        if self.predict_average_layers: # predict layer
                            if i == 0:
                                hw_ent_temp = highway_entropy.cpu().numpy()[0]
                            else:
                                hw_ent_temp = hw_ent_temp + highway_entropy.cpu().numpy()[0]

                    if (i == self.predict_layer - 1): # predict layer

                        if highway_entropy < ent_:
                            new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                            raise HighwayException(new_output, i+1)

                        if self.predict_average_layers:
                            if i == 0:
                                hw_ent_temp = highway_entropy.cpu().numpy()[0]
                            else:
                                hw_ent_temp = hw_ent_temp + highway_entropy.cpu().numpy()[0]
                            hw_ent = hw_ent_temp / float((i+1))
                        else:
                            hw_ent = highway_entropy.cpu().numpy()[0]

                        #hash into lookup table w/ highway_entropy
                        idx = (np.abs(self.lookup_table[:,0] - hw_ent)).argmin()
                        entropy_layers = np.transpose(self.lookup_table[idx,1:])
                        below_thresh = entropy_layers < ent_
                        k = np.argmax(below_thresh) # k is number of remaining layers
                        if (np.sum(below_thresh) == 0): #never hit threshold
                            k = entropy_layers.shape[0] - 1

                    # other layers (count down and then trigger highway exit if layer < self.num_hidden_layers)
                    elif ((i >= self.predict_layer) and (i < self.config.num_hidden_layers - 2)):

                        if (self.extra_layer):
                            if k == 0:
                                if highway_entropy < ent_:
                                    new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                                    raise HighwayException(new_output, i+1)
                            elif k==-1:
                                new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                                raise HighwayException(new_output, i+1)
                        else:
                            if (not self.no_ee_before):
                                if highway_entropy < ent_:
                                    new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                                    raise HighwayException(new_output, i+1)
                            if k == 0: #exit after counting down layers (CHECK CORRECT # OF LAYERS)
                                new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                                raise HighwayException(new_output, i+1)
                        k = k - 1
            else:
                all_highway_exits = all_highway_exits + (highway_exit,)

        #use this????
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        outputs = outputs + (all_highway_exits,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class AlbertModel(AlbertPreTrainedModel):

    def __init__(self, config, params):
        super().__init__(config, params)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.embeddings.requires_grad_(requires_grad=False)
        self.encoder = AlbertTransformer(config, params)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def init_highway_pooler(self):
        self.encoder.init_highway_pooler(self.pooler)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
            If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
            is a total of 4 different layers.

            These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
            while [2,3] correspond to the two inner groups of the second hidden layer.

            Any layer with in index other than [0,1,2,3] will result in an error.
            See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    #@add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Example::

        from transformers import AlbertModel, AlbertTokenizer
        import torch

        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        #CHECK THIS
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
         # add hidden_states and attentions if they are here
        return outputs

class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!


class AlbertHighway(nn.Module):
    r"""A module to provide a shortcut
    from
    the output of one non-final BertLayer in BertEncoder
    to
    cross-entropy computation in BertForSequenceClassification
    """
    def __init__(self, config):
        #super().__init__(config) ###
        super(AlbertHighway, self).__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        ##
        # self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs[0]
        # pooler_output = self.pooler(pooler_input)
        # "return" pooler_output

        #adding here:
        pooler_input = self.pooler(pooler_input[:,0])
        pooler_output = self.pooler_activation(pooler_input)

        # BertModel
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        # "return" bodel_output

        # Dropout and classification
        pooled_output = bmodel_output[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output



class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config, params)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.albert = AlbertModel(config, params)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    #@add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_layer=-1,
        train_highway=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        logits ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        Examples::

            from transformers import AlbertTokenizer, AlbertForSequenceClassification
            import torch

            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=labels)
            loss, logits = outputs[:2]

        """
        exit_layer = self.num_layers

        try:
            outputs = self.albert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        if not self.training:
            original_entropy = entropy(logits)
            highway_entropy = []
            highway_logits_all = []
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # work with highway exits
            highway_losses = []
            for highway_exit in outputs[-1]:
                highway_logits = highway_exit[0]
                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_entropy.append(highway_exit[2])
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                highway_losses.append(highway_loss)

            if train_highway:
                outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course
            else:
                outputs = (loss,) + outputs
        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) +\
                          (highway_logits_all[output_layer],) +\
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        #self.albert = AlbertModel(config)
        self.albert = AlbertModel(config, params)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_layer=-1,
        train_highway=False
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        end_scores: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        # The checkpoint albert-base-v2 is not fine-tuned for question answering. Please see the
        # examples/run_squad.py example to see how to fine-tune a model to a question answering task.

        from transformers import AlbertTokenizer, AlbertForQuestionAnswering
        import torch

        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertForQuestionAnswering.from_pretrained('albert-base-v2')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_dict = tokenizer.encode_plus(question, text, return_tensors='pt')
        start_scores, end_scores = model(**input_dict)

        """

        exit_layer = self.num_layers

        try:
            outputs = self.albert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + outputs[2:]

        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            start_logits = outputs[0]
            end_logits = outputs[1]

        if not self.training:
            # original_start_entropy = entropy(start_logits)
            # original_end_entropy = entropy(end_logits)
            original_entropy = entropy(logits)
            highway_entropy = []
            # highway_start_logits_all = []
            # highway_end_logits_all = []
            highway_logits_all = []

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs

            # work with highway exits
            highway_losses = []
            for highway_exit in outputs[-1]:
                highway_logits = highway_exit[0]
                highway_start_logits, highway_end_logits = highway_logits.split(1, dim=-1)
                highway_start_logits = highway_start_logits.squeeze(-1)
                highway_end_logits = highway_end_logits.squeeze(-1)

                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_entropy.append(highway_exit[1])

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(highway_start_logits, start_positions)
                end_loss = loss_fct(highway_end_logits, end_positions)
                highway_loss = (start_loss + end_loss) / 2
                highway_losses.append(highway_loss)

            if train_highway:
                outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course
            else:
                outputs = (total_loss,) + outputs

        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) +\
                          (highway_logits_all[output_layer],) +\
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
