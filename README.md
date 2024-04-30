# Llama2-finetune on Apple silicon (M3 max)

This is a guide to finetune the Llama2 model on Apple silicon (M3 max) using the [Hugging Face Transformers](https://huggingface.co/transformers/) library.

Training a model on Apple silicon is a little bit different from training on other platforms due to several reasons:

* need to use 'mps'
* bitsandbytes quantization configuration does not work on Apple silicon
* training optimizer 'paged_adamw_32bit' does not work on Apple silicon
* at time of writing, transformers==4.38.2 does not work on Apple silicon, need to use transformers==4.38.1
