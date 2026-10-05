"""Correct first-run padding counters without changing any model response.

The generation tensor was padded with EOS, whereas tokenizer.pad_token_id is
another token. Original counters therefore included some padding. Retokenize
visible generated text for descriptive counts only; exact EOS/truncation state
cannot be reconstructed without the original token IDs. Scoring never uses it.
"""
import json,sys
from pathlib import Path
from .file_model_comparison import OUT,MODEL,ROOT,write

def main():
    sys.path.insert(0,str(ROOT/'artifacts/comparison-runtime'))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    path=OUT/'qwen_predictions.jsonl';raw=path.read_text(encoding='utf-8')
    records=[json.loads(line) for line in raw.splitlines()]
    backup=OUT/'qwen_predictions.before_padding_metadata_fix.jsonl'
    if not backup.exists():backup.write_text(raw,encoding='utf-8')
    for row in records:
        if 'generated_tokens' in row:row['original_tensor_nonpad_count']=row.pop('generated_tokens')
        if 'truncated' in row:row['original_tensor_budget_flag']=row.pop('truncated')
        row['generated_text_tokens']=len(tokenizer.encode(row['response'],add_special_tokens=False))
        row['exact_generation_truncation_known']=False
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    runtime=json.loads((OUT/'qwen_runtime.json').read_text(encoding='utf-8'))
    if 'generated_tokens_this_run' in runtime:runtime['original_counter_including_some_padding']=runtime.pop('generated_tokens_this_run')
    runtime['generated_text_tokens']=sum(r['generated_text_tokens'] for r in records)
    runtime['cuda_synchronized_generation_seconds']=sum(records[i]['batch_seconds'] for i in range(0,len(records),12))
    runtime['amortized_seconds_per_request']=runtime['seconds']/len(records)
    runtime['token_metadata_note']='Visible generated text retokenized, no special tokens. Original tensor padding counters preserved for audit; exact truncation unknown. Responses, bindings, plans and timing unchanged.'
    write(OUT/'qwen_runtime.json',runtime)
    print(json.dumps({'rows':len(records),'responses_unchanged':True,'seconds':runtime['seconds'],'cuda_generation_seconds':runtime['cuda_synchronized_generation_seconds']}))

if __name__=='__main__':main()
