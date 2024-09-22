# Efficient and Effective Model Extraction
Code for the paper 'Efficient and Effective Model Extraction' submitted to IEEE ICASSP 2025.

This repository is still undergoing reorganization to provide continuous support for the paper.

---

## **Important Notice:**  
Due to an unfortunate LaTeX typo, there is a mistake in the formula presented in **Proposition** under Section **III.METHODOLOGY B.Temperature Scaling in Black-box Extraction**. We sincerely apologize for this oversight and all known typos have been corrected. The revised version of the paper can be found in the following document: [**/Document/E3_Revision.pdf**](./Document/E3_Revision.pdf).

Thank you for your understanding and your interest in our work.

---

## Usage
The core implementation is now available. Experiment parameters on CIFAR-10 are listed in the corresponding `.py` files. We will provide more guidelines with flexible parameter settings in the coming days.

To run E3:

1. First, execute `query_selection.py` to generate the query set from the candidate out-of-distribution (OOD) set.
2. Then, run `model_extraction.py` to perform the extraction attack.
3. Finally, if desired, execute `distribution_alignment.py` to apply test-time distribution alignment.
