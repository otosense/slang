- [Slang: Light weight tools to build signal languages](#slang--light-weight-tools-to-build-signal-languages)
  * [A story to paint the horizon](#a-story-to-paint-the-horizon)
  * [Okay, but what does a pipeline look like in slang](#okay--but-what-does-a-pipeline-look-like-in-slang)
- [Sound Language](#sound-language)
- [Structural and Syntactical Pattern Recognition](#structural-and-syntactical-pattern-recognition)
- [Semantic Structure](#semantic-structure)
- [Acoustics Structure](#acoustics-structure)
  * [Alphabetization](#alphabetization)
  * [Snips network](#snips-network)
- [Snips Annotations](#snips-annotations)
  * [Relationship between Annotations and the Syntactic Approach](#relationship-between-annotations-and-the-syntactic-approach)
- [Modeling](#modeling)
- [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

`Slang` stands for "Signal LANGuage". It is meant to build signal ML systems that translate streams of signals to streams of interpretable and more importantly, actionable information. 

How is it similar and how is it different to other known approaches to this signal ML problem ? We'll try to clarify that here. 
The main idea is in the name: Signal **Language**. That is, the methods used gravitate around mapping the data too language-like data and using language models to solve problems. 

ML can be seen as purposeful information compression. Raw data, or signal, is structured and through a series of transformations (intermediate "feature vectors"), is converted into semantically interpretable, often actionable information. 

What's `slang`'s take on that?

Slang focuses on signals -- or more generally, on (usually, but not necessarily, timestamped) sequential data. 
The layers of transformation in a slang pipeline can be seen as incremental conversions of interval annotations. 
Every layer of annotations add structural and quantitative information, progressing from low level descriptions (e.g. acoustics) to hi level descriptions (e.g. a detection or classification). 

A noteworthy embodiment of this approach is the use of [vector quantization](https://en.wikipedia.org/wiki/Vector_quantization) codec to encode a previous layer's annotations into annotations, called snips, akin to words (or letters, phonemes, tokens) of a natural language. 
Where vector quantization (and most codecs) is aimed a enabling the original data to be decoded on the other side, the goal here is purposeful information compression, or even some general form of [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy). That is, we want to purposefully (1) keep the information that is needed to carry out the objectives of the system and, sometimes (2) not keep some specific types  of information (privacy).

The decoder acts as an _interpreter_ of the snips. It uses a codebook to _translate_ them into a sequence of annotations that are meaningful in the context the decoder is meant for. It's important to note that, unlike classical vector quantization, the codebook is not limited to decoding only one snip at a time into something approximating the original input, but rather _generating annotations based on potential large windows of streaming snips_. 

Lets give an illustrative example of what this "middle langauge" that snips offer does. 
[MIDI](https://www.cs.cmu.edu/~music/cmsip/readings/MIDI%20tutorial%20for%20programmers.html) is a standard for transmitting and storing music, initially made for digital synthesizers. It sends musical notes, timings, and pitch, *not recorded sounds*, allowing the receiver to play music using its sound library. 

MIDI captures only a minimal set of features that focus on the production of music. The decoder side can general the sound that will reproduce something resembling the original music. Being designed primarily around the conventions of western music (12-tone system, rational durations, etc.), it would be less appropriate for non-western music and would be almost incapable at encoding sounds in general.

What `slang` is designed to do is to learn such a signal codec as MIDI for a given target domain. 


You can read more about this in this [Ideas on `slang`](https://github.com/otosense/slang/wiki/Ideas-on-%60slang%60) wiki.
[21] C. Yu, D. H. Ballard, “On the integration of grounding language and learning objects”, AAAI, 2004. 

[22] S-C. Zhu, D. Mumford, “A stochastic Grammar of Images”, 
Foundations and trends in Computer Vision and Graphics, 2006
