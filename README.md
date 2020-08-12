# Visualizing-attention-layer-in-text-classification-deep-learning-algorithim
I have finished creating a prototype algorithm on a small dataset for work that can detect the gradients of a newly created classification of Community Engaged Research. This model has a custom attention layer added in and I have found that visualizing this layer can actually help see what words are important and what words are not. Although there may be some skepticism on whether or not attention layers can in fact explain a model(SEE https://arxiv.org/abs/1902.10186), there are other articles out there that use this technique. I have found it to be very helpful, and it aligns with what I saw in the data when creating this new way of classifying Community Engaged Research Protocols. Hope you enjoy and please critique!!!

My evidence for why I believe the attention heat map works is simply because the words that the attention layer weighted of higher importance were words that we saw as important to a specific class when we classified them. For example, the second class is where researchers partner with an organization to conduct their research, but the extent of the partners' involvement was that they just put up recruitment flyers in their buildings(which is why they are a 2), and the word “recruitment” came up as an important weighted word for that example. Another example would be the fifth class where a researcher partners with an organization and that organization shares governance over decisions being made during the research process and they can even form an advisory board or committee, and one of the highest weighted words that came from the fifth class example was “committee”.

So I understand it is hard to explain how a model predicts but I do not believe this is a step in the wrong direction with explaining my model. 


# Acknowledgements:
Attention Viz was inspired by this StackOverflow post: 
- https://stackoverflow.com/questions/53867351/how-to-visualize-attention-weights

As well as these research papers: 
- https://arxiv.org/abs/1902.02181
- https://arxiv.org/abs/1805.12307

Code for attention layer was used from:
- https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/


# **Attention Layer Viz Example**:
![Alt text](attentionnotsorted.JPG?raw=true "Title")

#**Attention Layer**:
```python
from keras.engine.topology import Layer
from keras.layers import Layer
import keras.backend as K
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()
```
