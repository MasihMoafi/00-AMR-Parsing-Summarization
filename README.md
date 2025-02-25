# Semantic Graph-to-Text Summarization Implementation

This repository contains an independent implementation of a semantic graph-to-text summarization framework inspired by the paper:

**Text Summarization Based on Semantic Graphs: An Abstract Meaning Representation Graph-to-Text Deep Learning Approach**  
*Journal of Big Data (2024)*  
[https://doi.org/10.1186/s40537-024-00950-5](https://doi.org/10.1186/s40537-024-00950-5)

## Overview

This project implements a pipeline for abstractive text summarization that includes:
- **Semantic Graph Parsing:** Extracts AMR graphs from input sentences.
- **Graph Construction & Transformation:** Combines and linearizes sentence-level graphs into a unified representation.
- **Deep Learning Prediction:** Generates summaries using models such as an attentive seq2seq with pointer-generator and transformer-based architectures.
  
## Note
This codebase is an independent implementation inspired by the paper mentioned above. It is not the original work of the paper's authors.

## License
This project is licensed under the MIT License.
