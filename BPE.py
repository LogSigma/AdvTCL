import six
import functools
from collections import Counter
from collections import defaultdict

class KoreanHandler:
    
    def __init__(self):
        self.TYPE_INITIAL = 0x001
        self.TYPE_MEDIAL = 0x010
        self.TYPE_FINAL = 0x100
        self.INITIALS = list(map(six.unichr, [0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
                                  0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
                                  0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
                                  0x314e]))
        self.MEDIALS = list(map(six.unichr, [0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
                                 0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
                                 0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
                                 0x3161, 0x3162, 0x3163]))
        self.FINALS = list(map(six.unichr, [0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
                                0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
                                0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
                                0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
                                0x314c, 0x314d, 0x314e]))
        self.CHAR_LISTS = {self.TYPE_INITIAL: self.INITIALS,
                           self.TYPE_MEDIAL: self.MEDIALS,
                           self.TYPE_FINAL: self.FINALS}
        self.CHAR_SETS = dict(map(lambda x: (x[0], set(x[1])), six.iteritems(self.CHAR_LISTS)))
        self.CHARSET = functools.reduce(lambda x, y: x.union(y), self.CHAR_SETS.values(), set())
        self.CHAR_INDICES = dict(map(lambda x: (x[0], dict([(c, i) for i, c in enumerate(x[1])])),
                six.iteritems(self.CHAR_LISTS)))

        def check_syllable(self, x):
            return 0xAC00 <= ord(x) <= 0xD7A3

        def jamo_type(self, x):
            t = 0x000
            for type_code, jamo_set in six.iteritems(self.CHAR_SETS):
                if x in jamo_set:
                    t |= type_code
            
            return t

        def split_syllable_char(self, x):
            """
            Splits a given korean character into components.
            """
            if len(x) != 1:
                raise ValueError("Input string must have exactly one character.")

            if not check_syllable(x):
                raise ValueError(
                    "Input string does not contain a valid Korean character.")

            diff = ord(x) - 0xAC00
            _m = diff % 28
            _d = (diff - _m) // 28

            initial_index = _d // 21
            medial_index = _d % 21
            final_index = _m

            if not final_index:
                result = (self.INITIALS[initial_index], self.MEDIALS[medial_index])
            else:
                result = (
                    self.INITIALS[initial_index], self.MEDIALS[medial_index],
                    self.FINALS[final_index - 1])

            return result

        def join_jamos_char(self, initial, medial, final=None):
            """
            Combines jamos to produce a single syllable.
            """
            if initial not in self.CHAR_SETS[self.TYPE_INITIAL] or medial not in self.CHAR_SETS[
                self.TYPE_MEDIAL] or (final and final not in self.CHAR_SETS[self.TYPE_FINAL]):
                raise ValueError("Given Jamo characters are not valid.")

            initial_ind = self.CHAR_INDICES[self.TYPE_INITIAL][initial]
            medial_ind = self.CHAR_INDICES[self.TYPE_MEDIAL][medial]
            final_ind = self.CHAR_INDICES[self.TYPE_FINAL][final] + 1 if final else 0

            b = 0xAC00 + 28 * 21 * initial_ind + 28 * medial_ind + final_ind

            result = six.unichr(b)

            assert self.check_syllable(result)

            return result

        def split_syllables(self, string):
            """
            Splits a sequence of Korean syllables to produce a sequence of jamos.
            Irrelevant characters will be ignored.
            """
            new_string = ""
            for c in string:
                if not check_syllable(c):
                    new_c = c
                else:
                    new_c = "".join(split_syllable_char(c))
                new_string += new_c

            return new_string

        def join_jamos(self, string):
            """
            Combines a sequence of jamos to produce a sequence of syllables.
            Irrelevant characters will be ignored.
            """
            last_t = 0
            queue = []
            new_string = ""

            def flush(queue, n=0):
                new_queue = []

                while len(queue) > n:
                    new_queue.append(queue.pop())

                if len(new_queue) == 1:
                    result = new_queue[0]
                elif len(new_queue) >= 2:
                    try:
                        result = join_jamos_char(*new_queue)
                    except ValueError:
                        # Invalid jamo combination
                        result = "".join(new_queue)
                else:
                    result = None

                return result

            for c in string:
                if c not in self.CHARSET:
                    if queue:
                        new_c = flush(queue) + c
                    else:
                        new_c = c

                    last_t = 0
                else:
                    t = jamo_type(c)
                    new_c = None

                    if t & self.TYPE_FINAL == self.TYPE_FINAL:
                        if not (last_t == self.TYPE_MEDIAL):
                            new_c = flush(queue)
                    elif t == self.TYPE_INITIAL:
                        new_c = flush(queue)
                    elif t == self.TYPE_MEDIAL:
                        if last_t & self.TYPE_INITIAL == self.TYPE_INITIAL:
                            new_c = flush(queue, 1)
                        else:
                            new_c = flush(queue)

                    last_t = t
                    queue.insert(0, c)

                if new_c:
                    new_string += new_c

            if queue:
                new_string += flush(queue)

            return new_string

class BPE:
    
    def __init__(self, n_iters=100, verbose=True, encoding='utf-8', consonants=False):
        self.n_iters = n_iters if n_iters > 0 else 100
        self.units = {}
        self.max_length = 0
        self.verbose = verbose
        self.encoding = encoding
        self.consonants = consonants
        
    def train(self, sents):
        if self.verbose:
            print('begin vocabulary scanning', end='', flush=True)
        
        vocabs = self._sent_to_vocabs(sents)
        if self.verbose:
            print('\rterminated vocabulary scanning', flush=True)
        
        self.units = self._build_subword_units(vocabs)
    
    def _sent_to_vocabs(self, sents):        
        vocabs = Counter((eojeol.replace('_', '') for sent in sents for eojeol in sent.split() if eojeol))
        return {' '.join(w) + ' _': c for w,c in vocabs.items() if w}
        
    def _build_subword_units(self, vocabs):
        def get_stats(vocabs):
            pairs = defaultdict(int)
            for word, freq in vocabs.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[(symbols[i],symbols[i+1])] += freq
            return pairs
        
        def merge_vocab(pair, v_in):
            v_out = {}
            bigram = ' '.join(pair)
            replacer = ''.join(pair)
            for word, freq in v_in.items():
                w_out = word.replace(bigram, replacer)
                v_out[w_out] = freq
            return v_out
        
        for i in range(self.n_iters + 1):
            pairs = get_stats(vocabs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocabs = merge_vocab(best, vocabs)
            if self.verbose and i % 100 == 99:
                print('\rtraining bpe {} / {}'.format(i+1, self.n_iters), end='', flush=True)
        if self.verbose:
            print('\rtraining bpe was done{}'.format(' '*40))
        
        units = {}
        for word, freq in vocabs.items():
            for unit in word.split():
                units[unit] = units.get(unit, 0) + freq
        self.max_length = max((len(w) for w in units))
        return units
    
    def tokenize(self, s):
        return ' '.join([self._tokenize(w) for w in s.split()])
    
    def _tokenize(self, w):
        def initialize(w):
            w += '_'
            subwords = []
            n = len(w)
            for b in range(n):
                for e in range(b+1, min(n, b+self.max_length)+1):
                    subword = w[b:e]
                    if not subword in self.units:
                        continue
                    subwords.append((subword, b, e, e-b))
            return subwords
        
        def longest_match(subwords):
            matched = []
            subwords = sorted(subwords, key=lambda x:(-x[3], x[1]))
            while subwords:
                s, b, e, l = subwords.pop(0) # str, begin, end, length
                matched.append((s, b, e, l))
                removals = []
                for i, (_, b_, e_, _) in enumerate(subwords):
                    if (b_ < e and b < e_) or (b_ < e and e_ > b):
                        removals.append(i)
                for i in reversed(removals):
                    del subwords[i]
            return sorted(matched, key=lambda x:x[1])
        
        subwords = initialize(w)
        subwords = longest_match(subwords)
        subwords = ' '.join([s for s, _, _, _ in subwords])
        return subwords
    
    def file_load(self, frame, iter_sent=False):
        with open(fname, encoding=self.encoding) as f:
            if iter_sent:
                for doc in f:
                    for sent in doc.split('  '):
                        yield sent
            else:
                for doc in f:
                    yield doc.strip()
                    
    def save(self, fname):
        with open(fname, 'w', encoding=self.encoding) as f:
            f.write('n_iters={}\n'.format(self.n_iters))
            f.write('max_length={}\n'.format(self.max_length))
            for unit, frequency in sorted(self.units.items(), key=lambda x:(-x[1], -len(x[0]))):
                f.write('{}\t{}\n'.format(unit, frequency))
                
    def load(self, fname):
        with open(fname, encoding=self.encoding) as f:
            try:
                self.n_iters = int(next(f).strip().split('=')[1])
                self.max_length = int(next(f).strip().split('=')[1])
            except Exception as e:
                print(e)
            
            self.units = {}
            for row in f:
                try:
                    unit, frequency = row.strip().split('\t')
                    self.units[unit] = int(frequency)
                except Exception as e:
                    print('BPE load exception: {}'.format(str(e)))
                    break