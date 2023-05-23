# Acoustic Stereotypy
Towards measuring Stereotypy in animal calls

This script is heavily based on the code developped for the Template matching baseline at DCASE Challenge task Few Shot Bioacoustic event detection. here:  https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/baselines/cross_correlation and is part of our recent publication:  "**Learning to detect an animal sound from five examples**" - https://arxiv.org/abs/2305.13210

## Description

A metric to evaluate similarity between sound events is needed in order to analyse how stereotyped are the vocalisations.

Here, similarity between two events is defined by the maximum value of their cross correlation. i.e : 
$sim(t,e) = max_{k} [xcorr(stft_t, stft_e(k:k+L))] $


 where $stft_t$ is the short term fourier transform of the template event(STFT), and $stft_e(k:k+L)$ is a slice of the STFT of a POS event e; k being the starting time index and L being the duration of the template event t in STFT frames.

. 
The first step consists in selecting the "template" events:  these are a random selection of 10 POS events across the whole audio recording.
Each of the template events is then cross-correlated with 30 randomly selected POS events. The average of the maximum cross correlation across the 30 operations results in a single value representing the average similarity between each template event and the remaining POS events in the audio file.
The final step is to average again this similarity value across all templates. Formally, these operation can be written as: 

$\frac{1}{T}\sum_{t}^{T}\frac{1}{E}\sum_{e}^{E}sim(t,e), $

 where T is the number of template events and E is the number of POS events randomly selected (30 in this implementation)

This proposed metric to measure similarity presents some limitations, namely events that differ from the templates on the time domain will be overly penalized, while a human annotator might still consider them to belong to the same class. A common example is when events present a similar pattern except that they differ in duration or because they are time-stretched.

Finally, when comparing stereotypy values across different classes, it is important to note the different granularity that these labels represent. As it is expected classes representing a specific call type or even calls from a single individual should have higher stereotypy values than broader classes. The results of these comparisons across different datasets are thus limited.

