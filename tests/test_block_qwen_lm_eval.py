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


def test_truncate_at_earliest_stop():
  text = 'answer\nif __name__ later\nclass Final'
  assert truncate_at_stops(text, ['\nclass ', '\nif __name__']) == 'answer'


def test_uniform_likelihood_fails_instead_of_reporting_masked_ce():
  try:
    require_masked_likelihood('uniform')
  except NotImplementedError as error:
    assert 'invalid for the uniform arm' in str(error)
  else:
    raise AssertionError('uniform likelihood must not report masked CE')
