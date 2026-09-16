from types import SimpleNamespace

from discrete_diffusion.evaluations.block_qwen_eval_utils import (
    encode_context_continuation,
    generation_request_args,
    require_masked_likelihood,
    truncate_at_stops,
)


def test_generation_request_args_preserve_task_limits_and_stops():
  request = SimpleNamespace(args=(
      'prompt',
      {'max_gen_toks': 1024, 'until': ['\nclass ', '\nif __name__']},
  ))
  limit, stops = generation_request_args(request, default_max_tokens=512)
  assert limit == 1024
  assert stops == ['\nclass ', '\nif __name__']


def test_generation_request_args_support_aliases_and_none_stops():
  request = SimpleNamespace(args=('prompt', {
      'max_new_tokens': 37,
      'until': None,
  }))
  assert generation_request_args(request, 512) == (37, [])


def test_encode_pair_moves_trailing_space_and_seeds_empty_context():
  class TinyTokenizer:
    bos_token_id = 99
    eos_token_id = 100

    def __call__(self, text, add_special_tokens=False):
      del add_special_tokens
      table = {
          '': [],
          'Answer:': [1, 2],
          'Answer: A': [1, 2, 3],
          'A': [4],
          'The': [10],
          'There': [99],
          're': [11],
      }
      return {'input_ids': table[text]}

  tok = TinyTokenizer()
  assert encode_context_continuation(tok, 'Answer: ', 'A') == ([1, 2], [3])
  assert encode_context_continuation(tok, '', 'A') == ([99], [4])
  # BPE merge: joint 'There' is not a prefix of 'The' → fall back to 're'.
  assert encode_context_continuation(tok, 'The', 're') == ([10], [11])
  # ChatML ends with newline; must NOT steal it onto the continuation
  # (that made every MMLU choice share first token \\n → chance).
  class ChatMLTok(TinyTokenizer):
    def __call__(self, text, add_special_tokens=False):
      del add_special_tokens
      table = {
          '<|im_start|>assistant\n': [7, 8, 9],
          '<|im_start|>assistant\n A': [7, 8, 9, 3],
          '<|im_start|>assistant\n\n A': [7, 8, 9, 9, 3],
          ' A': [3],
          '\n A': [9, 3],
      }
      return {'input_ids': table[text]}

  chat = ChatMLTok()
  assert encode_context_continuation(
      chat, '<|im_start|>assistant\n', ' A') == ([7, 8, 9], [3])


def test_truncate_at_earliest_stop():
  text = 'answer\nif __name__ later\nclass Final'
  assert truncate_at_stops(text, ['\nclass ', '\nif __name__']) == 'answer'


def test_humaneval_chat_stops_drop_def_truncation():
  from discrete_diffusion.evaluations.block_qwen_eval_utils import (
      assemble_humaneval_prediction,
      extract_fenced_code,
      humaneval_until_for_chat,
      prepare_code_completion,
      truncate_at_stops,
  )

  stock = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
  safe = humaneval_until_for_chat(stock)
  assert '\ndef' not in safe
  assert '<|im_end|>' in safe
  rewrite = '```python\ndef add(x, y):\n    return x + y\n```'
  assert truncate_at_stops(rewrite, stock) == '```python'
  assert 'def add' in truncate_at_stops(rewrite, safe)

  body = prepare_code_completion(rewrite, 'humaneval')
  assert body.startswith('def add')
  prompt = 'def add(x, y):\n    """Add."""\n'
  prog = assemble_humaneval_prediction(prompt, rewrite, 'add')
  assert prog.count('def add') == 1
  assert 'return x + y' in prog
  # Body-only completion still concatenates.
  prog2 = assemble_humaneval_prediction(prompt, '    return x - y\n', 'add')
  assert prog2 == prompt + '    return x - y\n'
  assert extract_fenced_code('no fence') is None


def test_uniform_likelihood_fails_instead_of_reporting_masked_ce():
  try:
    require_masked_likelihood('uniform')
  except NotImplementedError as error:
    assert 'invalid for the uniform arm' in str(error)
  else:
    raise AssertionError('uniform likelihood must not report masked CE')


def test_patch_code_eval_metric_cache_injects_unique_experiment_id(monkeypatch):
  import evaluate as hf_evaluate
  from discrete_diffusion.evaluations.block_qwen_lm_eval import (
      patch_code_eval_metric_cache,
  )

  captured = {}

  def fake_load(path, *args, **kwargs):
    captured['path'] = path
    captured['kwargs'] = dict(kwargs)
    return 'metric'

  monkeypatch.setattr(hf_evaluate, 'load', fake_load)
  # Allow re-patch after a prior test/import may have patched once.
  if getattr(hf_evaluate.load, '_uni_d2_code_eval_patched', False):
    delattr(hf_evaluate.load, '_uni_d2_code_eval_patched')
  monkeypatch.setenv('SLURM_JOB_ID', 'jobX')
  monkeypatch.setenv('RANK', '3')
  monkeypatch.setenv('OUT_DIR', '/tmp/uni_d2_code_eval_test')
  patch_code_eval_metric_cache()
  assert hf_evaluate.load('code_eval') == 'metric'
  assert captured['path'] == 'code_eval'
  eid = captured['kwargs']['experiment_id']
  assert eid.startswith('code_eval_jobX_r3_')
  assert captured['kwargs']['keep_in_memory'] is True
  assert eid in captured['kwargs']['cache_dir']
  # Non-code metrics must pass through unchanged.
  captured.clear()
  assert hf_evaluate.load('accuracy') == 'metric'
  assert 'experiment_id' not in captured['kwargs']
