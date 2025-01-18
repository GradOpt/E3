# Efficient and Effective Model Extraction
Code for the paper 'Efficient and Effective Model Extraction' accepted at IEEE ICASSP 2025.

---

## Usage
The core implementation is now available. Experimental parameters on CIFAR-10 are listed in the corresponding `.py` files.

To run E3:

1. First, execute `query_selection.py` to generate the query set from the candidate out-of-distribution (OOD) set.
2. Then, run `model_extraction.py` to perform the extraction attack.
3. Finally, if desired, execute `distribution_alignment.py` to apply test-time distribution alignment.
