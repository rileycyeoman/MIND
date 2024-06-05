# MIND
Mental health Identification using No-label Distillation

## Overview
MIND is a machine learning model that combines META's DINO model and a Long Short-Term Memory (LSTM) network. This research seeks to classify mental states that are more time-variant in nature. This has been a challenge in past literature for a multitude of reasons:
- Previous works only sought to classify based on a set of 7-9 basic emotions (datasets only provide so much information). These labels are too broad to accurately describe something as complex as depression or anxiety. They do not encompass the entire emotion spectrum, nor do they capture the nuance of complex emotion. Common labels include:
    - $\textcolor{red}{Anger}$
    - $\textcolor{orange}{Surprise}$
    - $\textcolor{yellow}{Happy}$
    - $\textcolor{lime}{Fear}$
    - $\textcolor{blue}{Sad}$
    - $\textcolor{orchid}{Disgust}$
    - $\textcolor{purple}{Contempt}$
    - $\textcolor{gray}{Neutral}$  

- Emotions are defined differently from greater psychological issues. Being either happy and anxious can both be short-term psychological symptoms, however, something like Generalized Anxiety Disorder can't necessarily be seen as instantaneous. Dually, "happy," in terms of an image or video classification, can be from a single frame or set of images being strong enough probability-wise to force a machine learning model to interpret the data as a single emotion. 


## Previous Work
### DINO

### TimeSformer


## Model

## How to Use

```
> git clone https://github.com/rileycyeoman/MIND.git
```



