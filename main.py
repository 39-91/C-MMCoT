import huggingface_hub
import numpy as np
import sys
print(f"Loaded huggingface_hub version: {huggingface_hub.__version__}")
print(f"huggingface_hub path: {huggingface_hub.__file__}")
print(f"Python sys.path: {sys.path}")
import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from model import T5ForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_img, ScienceQADatasetStd, ScienceQADatasetImg
from utils_prompt import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
import numpy as np # <--- Make sure that numpy is imported
import numpy.core.multiarray

safe_globals_to_add = [
    numpy.core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,
    np.uint32,
    np.int64,
    np.float32,
    np.bool_,
    np.int32,
    np.float64
]

try:
    torch.serialization.add_safe_globals(safe_globals_to_add)
    print("INFO: Added potentially unsafe globals to torch serialization allowlist.")
except AttributeError:
    print("WARNING: torch.serialization.add_safe_globals not found (older PyTorch?)")
except Exception as e:
    print(f"WARNING: Failed to add safe globals: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=512)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    parser.add_argument('--use_generate', default=True, action='store_true', help='only for baseline to improve inference speed')
    # parser.add_argument('--ignore_pad_token_for_loss', default=True, action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    # parser.add_argument('--user_msge', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--user_msg', type=str, default="rationale", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default="clip", choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='The checkpoint directory to resume training from.')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', default=True, action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
        
def T5Trainer(
    dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']


    # <<<--- Add start --->>>
    # Set the number of samples you want to run
    # N_SAMPLES_FOR_TESTING = 1000
    # if args.evaluate_dir is not None: # Only limit the sample size in the reasoning mode during training
    #     console.log(f"[!] DEBUG MODE: Limiting validation and test sets to first {N_SAMPLES_FOR_TESTING} samples.")
    #     if len(val_qids) > N_SAMPLES_FOR_TESTING:
    #          val_qids = val_qids[:N_SAMPLES_FOR_TESTING]
    #     if len(test_qids) > N_SAMPLES_FOR_TESTING:
    #          test_qids = test_qids[:N_SAMPLES_FOR_TESTING]
    #     console.log(f"  - Using {len(val_qids)} samples for validation set.")
    #     console.log(f"  - Using {len(test_qids)} samples for test set.")
    # else:
    #      console.log("[!] Running in Training mode or without --evaluate_dir. Using full dataset splits.")
    # <<<--- Over --->>>

    # First, determine the final output directory based on whether it is an evaluation mode or not
    if args.evaluate_dir is not None:  # Evaluation/Prediction Patterns
        # We want to output to a subdirectory under args.output_dir

        evaluated_model_name_base = os.path.basename(args.evaluate_dir.strip('\\/'))
        if not evaluated_model_name_base:
            evaluated_model_name_base = "unknown_model"

        # Use the args.output_dir (the output base path you specify for this prediction run on the command line)
        # Under this base path, create a subdirectory that uniquely identifies the prediction
        predict_run_name = f"predict_output_for_{evaluated_model_name_base}_{args.user_msg}_{args.prompt_format}"
        save_dir = os.path.join(args.output_dir, predict_run_name)  # <--- The final save_dir

    else:  # Training mode
        # Keep your original training pattern save_dir logic
        # model_name_for_train_dir = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir_base_train = args.output_dir
        train_run_name = f"{args.user_msg}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        save_dir = os.path.join(save_dir_base_train, train_run_name)  # <--- The final save_dir

        # Make sure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  #  Use makedirs and allow already existing

    print(f"INFO: Final outputs (checkpoints/predictions) will be saved to: {save_dir}")  # Update the print information

    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size) 
        name_maps = dataframe['name_maps'] 
        image_features = dataframe['image_features']
        train_set = ScienceQADatasetImg(
            problems,
            train_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
        )
        eval_set = ScienceQADatasetImg(
            problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
        )
        test_set = ScienceQADatasetImg(
            problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model) 
        train_set = ScienceQADatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = ScienceQADatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )
        
        test_set = ScienceQADatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    # datacollator = DataCollatorForSeq2Seq(tokenizer)
    datacollator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    print("model parameters: ", model.num_parameters())
    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)
        
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED" 
        return answer  

    # accuracy for answer inference

    def compute_metrics_acc(eval_preds):
        # When predict_with_generate=True , eval_preds.predictions is token IDs
        # eval_preds.label_ids are real label token IDs
        preds_raw, targets_raw = eval_preds.predictions, eval_preds.label_ids

        # If preds_raw is a tuple (although for predict_with_generate=True it is usually not logits)
        if isinstance(preds_raw, tuple):
            preds_raw = preds_raw[0]

        # --- START: Add code that handles -100 and negative values ---
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0  # Fallback

        # --- Treatment preds_raw ---
        preds_processed = preds_raw
        if not isinstance(preds_processed, np.ndarray):
            try:
                preds_processed = np.array(preds_processed)
            except Exception as e:
                print(f"ERROR (compute_metrics_acc): Could not convert preds_raw to numpy array: {e}")
                return {'accuracy': 0.0}

        if preds_processed.size > 0:
            print(f"DEBUG (compute_metrics_acc): Min value in original preds: {preds_processed.min()}")
            preds_processed = np.where(preds_processed == -100, pad_token_id, preds_processed)
            preds_processed = np.clip(preds_processed, 0, None)
            preds_processed = preds_processed.astype(np.int64)
            print(f"DEBUG (compute_metrics_acc): Min value in processed preds: {preds_processed.min()}")
        else:
            print("DEBUG (compute_metrics_acc): preds_raw is empty.")
            return {'accuracy': 0.0}

        # --- Treatment targets_raw ---
        targets_processed = targets_raw
        if not isinstance(targets_processed, np.ndarray):
            try:
                targets_processed = np.array(targets_processed)
            except Exception as e:
                print(f"ERROR (compute_metrics_acc): Could not convert targets_raw to numpy array: {e}")
                return {'accuracy': 0.0}

        if targets_processed.size > 0:
            print(f"DEBUG (compute_metrics_acc): Min value in original targets: {targets_processed.min()}")
            targets_processed = np.where(targets_processed == -100, pad_token_id, targets_processed)
            targets_processed = targets_processed.astype(np.int64)
            print(f"DEBUG (compute_metrics_acc): Min value in processed targets: {targets_processed.min()}")
        else:
            print("DEBUG (compute_metrics_acc): targets_raw is empty.")
            return {'accuracy': 0.0}
        # --- END: Add code that handles -100 and negative values ---

        # Decode using the processed token ID
        try:
            preds_decoded = tokenizer.batch_decode(preds_processed, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True)
            targets_decoded = tokenizer.batch_decode(targets_processed, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
        except OverflowError as e:
            print(f"ERROR during batch_decode in compute_metrics_acc: {e}")
            print(
                f"  preds_processed min: {preds_processed.min()}, max: {preds_processed.max()}, dtype: {preds_processed.dtype}")
            print(
                f"  targets_processed min: {targets_processed.min()}, max: {targets_processed.max()}, dtype: {targets_processed.dtype}")
            return {'accuracy': 0.0}

        correct = 0
        assert len(preds_decoded) == len(targets_decoded)
        for idx, pred_text in enumerate(preds_decoded):
            reference_text = targets_decoded[idx]
            reference_ans = extract_ans(reference_text)  # extract_ans Requires the ability to process text
            extract_pred_ans = extract_ans(pred_text)  # extract_ans Requires the ability to process text

            if reference_ans == extract_pred_ans and reference_ans != "FAILED":  # Make sure the extracted answer is valid
                correct += 1
        return {'accuracy': 1.0 * correct / len(targets_decoded) if len(targets_decoded) > 0 else 0.0}
    
    # rougel for rationale generation
    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        # Get predictions and label_ids directly from eval_preds
        preds_token_ids, targets_token_ids = eval_preds.predictions, eval_preds.label_ids  # Use clearer variable names

        print(
            f"DEBUG: Original preds_token_ids shape: {preds_token_ids.shape if hasattr(preds_token_ids, 'shape') else 'N/A'}, dtype: {preds_token_ids.dtype if hasattr(preds_token_ids, 'dtype') else 'N/A'}")
        print(
            f"DEBUG: Original targets_token_ids shape: {targets_token_ids.shape if hasattr(targets_token_ids, 'shape') else 'N/A'}, dtype: {targets_token_ids.dtype if hasattr(targets_token_ids, 'dtype') else 'N/A'}")

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            print("WARNING: tokenizer.pad_token_id is None. Using 0 as a fallback for replacing -100.")
            pad_token_id = 0

        # --- Handel preds_token_ids ---
        if not isinstance(preds_token_ids, np.ndarray):
            try:
                preds_token_ids = np.array(preds_token_ids)
            except Exception as e:
                print(f"ERROR: Could not convert preds_token_ids to numpy array: {e}")
                return {}
        if preds_token_ids.size > 0:
            print(f"DEBUG: Min value in original preds_token_ids: {preds_token_ids.min()}")
            preds_token_ids = np.where(preds_token_ids == -100, pad_token_id, preds_token_ids)
            preds_token_ids = np.clip(preds_token_ids, 0, None)
            print(f"DEBUG: Min value in processed preds_token_ids: {preds_token_ids.min()}")
            preds_token_ids = preds_token_ids.astype(np.int64)  # Make sure it's an integer
        else:
            print("DEBUG: preds_token_ids is empty. Returning empty metrics.")
            return {}

        # --- Handel targets_token_ids ---
        if not isinstance(targets_token_ids, np.ndarray):
            try:
                targets_token_ids = np.array(targets_token_ids)
            except Exception as e:
                print(f"ERROR: Could not convert targets_token_ids to numpy array: {e}")
                return {}
        if targets_token_ids.size > 0:
            print(f"DEBUG: Min value in original targets_token_ids: {targets_token_ids.min()}")
            targets_token_ids = np.where(targets_token_ids == -100, pad_token_id, targets_token_ids)
            print(f"DEBUG: Min value in processed targets_token_ids: {targets_token_ids.min()}")
            targets_token_ids = targets_token_ids.astype(np.int64)  # Make sure it's an integer
        else:
            print("DEBUG: targets_token_ids is empty. Returning empty metrics.")
            return {}

        print(
            f"DEBUG: Final preds_token_ids before batch_decode (sample): {preds_token_ids[0, :20] if preds_token_ids.ndim > 1 and preds_token_ids.shape[0] > 0 and preds_token_ids.shape[1] > 20 else (preds_token_ids[0] if preds_token_ids.ndim > 0 and preds_token_ids.shape[0] > 0 else preds_token_ids)}")
        print(
            f"DEBUG: Final targets_token_ids before batch_decode (sample): {targets_token_ids[0, :20] if targets_token_ids.ndim > 1 and targets_token_ids.shape[0] > 0 and targets_token_ids.shape[1] > 20 else (targets_token_ids[0] if targets_token_ids.ndim > 0 and targets_token_ids.shape[0] > 0 else targets_token_ids)}")

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # vvvvvvvvvvvvvvvvvvvvvv Add a decoding step vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        try:
            # The processed token ID is decoded into a text string
            decoded_preds_text = tokenizer.batch_decode(preds_token_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)
            decoded_labels_text = tokenizer.batch_decode(targets_token_ids, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
        except OverflowError as e:
            print(f"ERROR during batch_decode in compute_metrics_rougel: {e}")
            print(
                f"  preds_token_ids min: {preds_token_ids.min()}, max: {preds_token_ids.max()}, dtype: {preds_token_ids.dtype}")
            print(
                f"  targets_token_ids min: {targets_token_ids.min()}, max: {targets_token_ids.max()}, dtype: {targets_token_ids.dtype}")
            return {}
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # ROUGE calculations are now performed using the decoded text
        try:
            # Try using globally defined postprocess_text and metric directly
            # Note that the variable names used here are the decoded_preds_text and decoded_labels_text decoded above
            processed_preds, processed_labels = postprocess_text(decoded_preds_text, decoded_labels_text)
            rouge_results = metric.compute(predictions=processed_preds, references=processed_labels)

            result = {}
            for key in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
                if key in rouge_results:
                    result[key] = rouge_results[key] * 100
                else:
                    print(
                        f"WARNING (compute_metrics_rougel): Metric key '{key}' not found in rouge_results from metric.compute().")

            if 'rougeL' not in result and 'rougeLsum' in result:
                result['rougeL'] = result['rougeLsum']
                print(f"INFO (compute_metrics_rougel): Using 'rougeLsum' ({result['rougeLsum']}) for 'rougeL'.")
            elif 'rougeL' not in result:
                print(
                    "CRITICAL WARNING (compute_metrics_rougel): 'rougeL' could not be computed or found. Setting to 0.0.")
                result['rougeL'] = 0.0

            result["gen_len"] = np.mean([len(pred.split()) for pred in processed_preds])  # Use the preds after postprocess
            print(f"DEBUG: Computed ROUGE metrics by compute_metrics_rougel: {result}")
            return result
        except NameError as e:
            print(
                f"WARNING (compute_metrics_rougel): Could not compute ROUGE due to NameError: {e}. Returning empty metrics.")
            return {}
        except Exception as e:
            print(
                f"ERROR (compute_metrics_rougel): An unexpected error occurred during ROUGE computation: {e}. Returning empty metrics.")
            return {}

    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=False,
            report_to="none",
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True if args.evaluate_dir is None else False,
            report_to="none",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel
    )

    if args.evaluate_dir is None:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(save_dir)
        
    metrics = trainer.evaluate(eval_dataset = test_set, max_length=args.output_len)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds_test, targets_test = predict_results.predictions, predict_results.label_ids
        else:

            preds_test = predict_results.predictions[0]
            targets_test = predict_results.label_ids
            preds_test = preds_test.argmax(axis=2)

        # --- START: Add code that handles -100 and negative values  ---
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0  # Fallback

        # --- Handle preds_test ---
        if not isinstance(preds_test, np.ndarray):
            try:
                preds_test = np.array(preds_test)
            except Exception as e:
                print(f"ERROR converting preds_test to numpy array: {e}")
                # Handle errors according to the actual situation

        if preds_test.size > 0:  # Make sure the array is not empty
            print(f"DEBUG T5Trainer (test_set): Min value in original preds_test: {preds_test.min()}")
            preds_test = np.where(preds_test == -100, pad_token_id, preds_test)
            preds_test = np.clip(preds_test, 0, None)
            preds_test = preds_test.astype(np.int64)
            print(f"DEBUG T5Trainer (test_set): Min value in processed preds_test: {preds_test.min()}")
        else:
            print("DEBUG T5Trainer (test_set): preds_test is empty.")

        # --- Handle targets_test ---
        if not isinstance(targets_test, np.ndarray):
            try:
                targets_test = np.array(targets_test)
            except Exception as e:
                print(f"ERROR converting targets_test to numpy array: {e}")
                # Handle errors according to the actual situation

        if targets_test.size > 0:  # Make sure the array is not empty
            print(f"DEBUG T5Trainer (test_set): Min value in original targets_test: {targets_test.min()}")
            targets_test = np.where(targets_test == -100, pad_token_id, targets_test)
            targets_test = targets_test.astype(np.int64)  # targets are usually already int64 and non-negative
            print(f"DEBUG T5Trainer (test_set): Min value in processed targets_test: {targets_test.min()}")
        else:
            print("DEBUG T5Trainer (test_set): targets_test is empty.")
        # --- END: Add code that handles -100 and negative values ---

        # Only now do the decoding
        preds = tokenizer.batch_decode(  # Use the processed preds_test
            preds_test, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(  # Use the processed targets_test
            targets_test, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results_ans = {}
        results_rationale = {}
        results_reference = {}
        
        num_fail = 0
        for idx, qid in enumerate(test_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            extract_pred = extract_ans(pred)
            if extract_pred != "FAILED":
                if extract_pred in args.options:
                    extract_pred = args.options.index(extract_pred)
                else:
                    extract_pred = random.choice(range(0,len(args.options)))
            else:
                num_fail += 1
                extract_pred = random.choice(range(len(args.options))) # random choose one option
            results_ans[str(qid)] = extract_pred
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref

        # Modified code
        data_file_path = os.path.join(args.data_root, "problems.json")  # Find problems.json directly under args.data_root
        print(f"DEBUG: Path to problems.json for get_scores: {data_file_path}")  # Add a print confirmation path
        scores = get_scores(results_ans, results_rationale, results_reference, data_file_path)
        preds = [pred.strip() for pred in preds]
        output_data = {
                "num_fail": num_fail,
                "scores": scores,
                "preds": preds,
                "labels": targets
        }
        output_prediction_file = os.path.join(save_dir,"predictions_ans_test.json")
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))

    # generate the rationale for the eval set
    if args.prompt_format == "QCM-LE" or args.prompt_format == "QCM-E":
        torch.cuda.empty_cache()
        # If the preds and targets variables are still in scope in the previous test_set processing, it is good to del them to avoid confusion
        # But to be safe, we use new variable names here like preds_eval_processed, targets_eval_processed

        predict_results_eval = trainer.predict(test_dataset=eval_set, max_length=args.output_len)
        if trainer.is_world_process_zero():
            if args.use_generate:
                preds_eval_raw, targets_eval_raw = predict_results_eval.predictions, predict_results_eval.label_ids
            else:

                preds_eval_raw = predict_results_eval.predictions[0]
                targets_eval_raw = predict_results_eval.label_ids
                preds_eval_raw = preds_eval_raw.argmax(axis=2)

            # --- START: Add code that handles -100 and negative values (similar to when processing test_set results ---
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0  # Fallback

            # --- Treatment preds_eval_raw ---
            preds_eval_processed = preds_eval_raw  # Assign values first, then process
            if not isinstance(preds_eval_processed, np.ndarray):
                try:
                    preds_eval_processed = np.array(preds_eval_processed)
                except Exception as e:
                    print(f"ERROR converting preds_eval_raw to numpy array: {e}")
                    # Handle errors according to the actual situation

            if preds_eval_processed.size > 0:  # Make sure the array is not empty
                print(f"DEBUG T5Trainer (eval_set): Min value in original preds_eval_raw: {preds_eval_processed.min()}")
                preds_eval_processed = np.where(preds_eval_processed == -100, pad_token_id, preds_eval_processed)
                preds_eval_processed = np.clip(preds_eval_processed, 0, None)
                preds_eval_processed = preds_eval_processed.astype(np.int64)
                print(f"DEBUG T5Trainer (eval_set): Min value in processed preds_eval: {preds_eval_processed.min()}")
            else:
                print("DEBUG T5Trainer (eval_set): preds_eval_raw is empty.")

            # --- 处理 targets_eval_raw ---
            targets_eval_processed = targets_eval_raw  # Assign values first, then process
            if not isinstance(targets_eval_processed, np.ndarray):
                try:
                    targets_eval_processed = np.array(targets_eval_processed)
                except Exception as e:
                    print(f"ERROR converting targets_eval_raw to numpy array: {e}")
                    # Handle errors according to the actual situation

            if targets_eval_processed.size > 0:  # Make sure the array is not empty
                print(
                    f"DEBUG T5Trainer (eval_set): Min value in original targets_eval_raw: {targets_eval_processed.min()}")
                targets_eval_processed = np.where(targets_eval_processed == -100, pad_token_id, targets_eval_processed)
                targets_eval_processed = targets_eval_processed.astype(np.int64)
                print(
                    f"DEBUG T5Trainer (eval_set): Min value in processed targets_eval: {targets_eval_processed.min()}")
            else:
                print("DEBUG T5Trainer (eval_set): targets_eval_raw is empty.")
            # --- END: Add code that handles -100 and negative values ---

            # Only now do the decoding, using the processed variables
            decoded_preds_eval = tokenizer.batch_decode(
                preds_eval_processed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            decoded_targets_eval = tokenizer.batch_decode(
                targets_eval_processed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # Subsequent codes use the decoded text
            processed_decoded_preds_eval = [pred.strip() for pred in decoded_preds_eval]
            output_data = {"preds": processed_decoded_preds_eval,  # Use processed and decoded preds
                           "labels": decoded_targets_eval}  # Use processed and decoded targets
            output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))
    

if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems':problems, 'qids':qids}

    T5Trainer(
        dataframe=dataframe,
        args = args
    )
