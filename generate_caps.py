import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

def _update_candidate(row, beam_size=20, omit_oov=True):
    prev_score = row['score']
    prev_seq = row['seq']
    next_token = row['token_score'].argsort()[::-1][:beam_size].astype(np.int32) + 1 # +1 for padding
    updated_score = prev_score + row['token_score'][next_token]
    updated_seq = list(np.concatenate([np.tile(prev_seq, (beam_size, 1)),
                                       np.atleast_2d(next_token).T],
                                      axis=1))
    return pd.DataFrame({'score': updated_score, 'seq': updated_seq})

def generate_caption(image, model, beam_size=20,
                     max_sentence_len=64, omit_oov=True,
                     oov_idx=1, bos_idx=2, eos_idx=3,
                     img_key='img_input', lang_key='lang_input'):
    """
    """
    padding = lambda x: pad_sequences(x, maxlen=max_sentence_len, padding='post')

    # predict first token
    batch = {img_key: np.atleast_2d(image), lang_key: padding(np.atleast_2d(bos_idx))}
    if len(model.output_shape) == 3:
        token_score = np.log(model.predict_on_batch(batch)[:,1,:])
    else:
        token_score = np.log(model.predict_on_batch(batch))
    first_tokens = {'seq': [[bos_idx, token + 1] for token in token_score[0].argsort()[::-1][:beam_size].astype(np.int32)],
                    'score': np.sort(token_score[0])[::-1][:beam_size]}
    candidate = pd.DataFrame(first_tokens)

    # predict succeeding tokens
    fixed_candidate = pd.DataFrame(columns=['score', 'seq', 'eos'])
    for i in range(2, max_sentence_len):
        batch = {img_key: np.tile(image, (beam_size, 1)), lang_key: padding(candidate['seq'])}
        if len(model.output_shape) == 3:
            token_score = np.log(model.predict_on_batch(batch)[:,i,:])
        else:
            token_score = np.log(model.predict_on_batch(batch))
        candidate['token_score'] = list(token_score)
        candidate = pd.concat(
            [_update_candidate(row[1], beam_size=beam_size, omit_oov=omit_oov) for row in candidate.iterrows()])
        candidate = candidate.sort_values('score', ascending=False)[:beam_size].reset_index(drop=True)

        ## store seq with EOS
        candidate['eos'] = [x[-1] == eos_idx for x in candidate['seq']]
        fixed_candidate = pd.concat([fixed_candidate, candidate[candidate['eos'] == True]])
        beam_size -= len(candidate[candidate['eos'] == True])
        if beam_size == 0:
            return fixed_candidate
        candidate = candidate[candidate['eos'] == False]

    return pd.concat([fixed_candidate, candidate])
