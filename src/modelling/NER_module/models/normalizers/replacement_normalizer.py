# from models.base_models import Normalizer
from src.modelling.NER_module.models.normalizers import cache


class MohaverekhanReplacementNormalizer():

    replacement_patterns1 = (
        (rf'([{cache.emojies}]+)(?=[ {cache.persians}{cache.punctuations}]|$)', r' EMOJI ', '', 0, 'mohaverekhan',
         'true'),
        (rf'({cache.email})(?=[ {cache.persians}{cache.punctuations}{cache.emojies}]|$)', r' EMAIL ', '', 0,
         'mohaverekhan', 'true'),
        (rf'({cache.link})(?=[ {cache.persians}{cache.punctuations}{cache.emojies}]|$)', r' LINK ', '', 0,
         'mohaverekhan', 'true'),
        (rf'({cache.id})(?=[ {cache.persians}{cache.emojies}]|$)', r' ID ', '', 0, 'mohaverekhan', 'true'),
        (rf'({cache.tag})(?=[ {cache.persians}{cache.punctuations}{cache.emojies}]|$)', r' TAG ', '', 0, 'mohaverekhan',
         'true'),
        (
            rf'({cache.num}|{cache.numf})(?=[ {cache.persians}{cache.num_punctuations}{cache.emojies}]|$)', r' NUMBER ',
            '',
            0, 'mohaverekhan', 'true'),
        (rf'(?<=[ {cache.persians}{cache.punctuations}])([{cache.emojies}]+)', r' EMOJI ', '', 0, 'mohaverekhan',
         'true'),
        (rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.email})', r' EMAIL ', '', 0,
         'mohaverekhan', 'true'),
        (
            rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.link})', r' LINK ', '', 0,
            'mohaverekhan',
            'true'),
        (rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.id})', r' ID ', '', 0, 'mohaverekhan',
         'true'),
        (rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.tag})', r' TAG ', '', 0, 'mohaverekhan',
         'true'),
        (rf'(?<=[ {cache.persians}{cache.num_punctuations}{cache.emojies}])({cache.num}|{cache.numf})', r' NUMBER ', '',
         0, 'mohaverekhan', 'true'),
        (r' +', r' ', 'remove extra spaces', 0, 'hazm', 'true'),
        # (r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F4CC\U0001F4CD]+', r' EMOJI ', 'emoji', 0, 'hazm', 'true'),
        # (r'[a-zA-Z0-9\._\+-]+@([a-zA-Z0-9-]+\.)+[A-Za-z]{2,}', r' EMAIL ', 'email', 0, 'hazm', 'true'),
        # (r'((https?|ftp):\/\/)?(?<!@)([wW]{3}\.)?(([\w-]+)(\.(\w){2,})+([-\w@:%_\+\/~#?&=]+)?)', r' LINK ', 'link, hazm + "="', 0, 'hazm', 'true'),
        # (r'([^\w\._]*)(@[\w_]+)([\S]+)', r' ID ', 'id', 0, 'hazm', 'true'),
        # (r'\#([\S]+)', r' TAG ', 'tag', 0, 'hazm', 'true'),
        # (r'-?[0-9۰۱۲۳۴۵۶۷۸۹]+([.,][0-9۰۱۲۳۴۵۶۷۸۹]+)?', r' NUMBER ', 'number', 0, 'mohaverekhan', 'true'),
    )
    replacement_patterns2 = (
        (rf'([{cache.emojies}]+)(?=[ {cache.persians}{cache.punctuations}]|$)', r' EMOJI ', '', 0, 'mohaverekhan',
         'true'),
        (rf'({cache.email})(?=[ {cache.persians}{cache.punctuations}{cache.emojies}]|$)', r' EMAIL ', '', 0,
         'mohaverekhan', 'true'),
        (rf'({cache.link})(?=[ {cache.persians}{cache.punctuations}{cache.emojies}]|$)', r' LINK ', '', 0,
         'mohaverekhan', 'true'),
        (rf'({cache.id})(?=[ {cache.persians}{cache.emojies}]|$)', r' ID ', '', 0, 'mohaverekhan', 'true'),
        # (rf'({cache.tag})(?=[ {cache.persians}{cache.punctuations}{cache.emojies}]|$)', r' TAG ', '', 0, 'mohaverekhan',
        #  'true'),
        (
            rf'({cache.num}|{cache.numf})(?=[ {cache.persians}{cache.num_punctuations}{cache.emojies}]|$)', r' NUMBER ',
            '',
            0, 'mohaverekhan', 'true'),
        (rf'(?<=[ {cache.persians}{cache.punctuations}])([{cache.emojies}]+)', r' EMOJI ', '', 0, 'mohaverekhan',
         'true'),
        (rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.email})', r' EMAIL ', '', 0,
         'mohaverekhan', 'true'),
        (
            rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.link})', r' LINK ', '', 0,
            'mohaverekhan',
            'true'),
        (rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.id})', r' ID ', '', 0, 'mohaverekhan',
         'true'),
        # (rf'(?<=[ {cache.persians}{cache.punctuations}{cache.emojies}])({cache.tag})', r' TAG ', '', 0, 'mohaverekhan', 'true'),
        (rf'(?<=[ {cache.persians}{cache.num_punctuations}{cache.emojies}])({cache.num}|{cache.numf})', r' NUMBER ', '',
         0, 'mohaverekhan', 'true'),
        (r' +', r' ', 'remove extra spaces', 0, 'hazm', 'true'),
        # (r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F4CC\U0001F4CD]+', r' EMOJI ', 'emoji', 0, 'hazm', 'true'),
        # (r'[a-zA-Z0-9\._\+-]+@([a-zA-Z0-9-]+\.)+[A-Za-z]{2,}', r' EMAIL ', 'email', 0, 'hazm', 'true'),
        # (r'((https?|ftp):\/\/)?(?<!@)([wW]{3}\.)?(([\w-]+)(\.(\w){2,})+([-\w@:%_\+\/~#?&=]+)?)', r' LINK ', 'link, hazm + "="', 0, 'hazm', 'true'),
        # (r'([^\w\._]*)(@[\w_]+)([\S]+)', r' ID ', 'id', 0, 'hazm', 'true'),
        # (r'\#([\S]+)', r' TAG ', 'tag', 0, 'hazm', 'true'),
        # (r'-?[0-9۰۱۲۳۴۵۶۷۸۹]+([.,][0-9۰۱۲۳۴۵۶۷۸۹]+)?', r' NUMBER ', 'number', 0, 'mohaverekhan', 'true'),
    )
    replacement_patterns1 = [(rp[0], rp[1]) for rp in replacement_patterns1]
    replacement_patterns2 = [(rp[0], rp[1]) for rp in replacement_patterns2]
    replacement_patterns1 = cache.compile_patterns(replacement_patterns1)
    replacement_patterns2 = cache.compile_patterns(replacement_patterns2)

    def normalize(self, text_content, keep_hashtag=False):

        # self.logger.info(f'>>> mohaverekhan_replacement_normalizer : \n{text_content}')

        # text_content = cache.normalizers['mohaverekhan-basic-normalizer']\
        #                 .normalize(text_content)
        # self.logger.info(f'>>> mohaverekhan-basic-normalizer : \n{text_content}')

        text_content = text_content.strip(' ')
        if keep_hashtag:
            replacement_patterns = self.replacement_patterns2

        else:
            replacement_patterns = self.replacement_patterns1
        for pattern, replacement in replacement_patterns:
            text_content = pattern.sub(replacement, text_content)
            # self.logger.info(f'> After {pattern} -> {replacement} : \n{text_content}')
        # self.logger.info(f'>> replace_text : \n{text_content}')
        text_content = text_content.strip()

        # self.logger.info(f"> (Time)({end_ts - beg_ts:.6f})")
        # self.logger.info(f'>>> Result mohaverekhan_replacement_normalizer : \n{text_content}')
        return text_content
