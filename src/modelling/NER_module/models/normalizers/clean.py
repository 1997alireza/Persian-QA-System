import re


# print("fjgjjg")


def clean_text(line):
    # print("fkgggggggggggggggj")
    # print(line)
    a = '\u200F'
    b = '\u200E'
    c = '\uFEFF'
    d = '\uFEFF'
    e = '\u200C'
    f = '\u200C'
    # new = ""
    # new = ~ s / a | b | c | d | e | f / _s_ / g
    line = line.replace(a, "_s_")
    line = line.replace(b, "_s_")
    line = line.replace(c, "_s_")
    line = line.replace(d, "_s_")
    line = line.replace(e, "_s_")
    line = line.replace(f, "_s_")
    g = '\u202A'
    h = '\u202B'
    i = '\u202C'
    j = '\u202D'
    k = '\u202E'
    line = line.replace(g, " ")
    line = line.replace(h, " ")
    line = line.replace(i, " ")
    line = line.replace(j, " ")

    # new = ~ s / g | h | i | j | k / / g

    aa = '\u2028'
    ab = '\u2029'
    #	ac = '\u202A'
    #	ad = '\u202B'
    #	ae = '\u202C'
    #	af = '\u202D'
    #	ag = '\u202E'
    aag = '\u0080'
    ah = '\u0081'
    ai = '\u0082'
    aj = '\u0083'
    ak = '\u0085'
    aak = '\u0086'
    aal = '\u0088'
    al = '\u0089'
    am = '\u008A'
    an = '\u008B'
    ao = '\u008C'
    ap = '\u008D'
    aq = '\u008E'
    aaq = '\u008F'
    ar = '\u0090'
    as_ = '\u0091'
    at = '\u0092'
    au = '\u0093'
    av = '\u0094'
    aw = '\u0095'
    ax = '\u0096'
    ay = '\u0097'
    az = '\u0098'
    ba = '\u0099'
    bb = '\u009A'
    bc = '\u009B'
    bd = '\u009C'
    be = '\u009D'
    bf = '\u009E'
    bg = '\u009F'
    bh = '\uFFFE'
    bi = '\uFDE1'

    # new = ~ s / aa | ab | ac | ad | ae | af | ag | aag | ah | ai | aj | ak | aak | aal | al | am
    # | an | ao | ap | aq | aaq | ar |as | at | au | av | aw | ax | ay | az | ba | bb | bc | bd | be | bf | bg // g
    line.replace(aa, "")
    line.replace(ab, "")
    # new.replace(ac, "")
    # new.replace(ad, "")
    # new.replace(ae, "")
    # new.replace(af, "")
    # new.replace(ag, "")
    line.replace(aag, "")
    line.replace(ah, "")
    line.replace(ai, "")
    line.replace(aj, "")
    line.replace(ak, "")
    line.replace(aak, "")
    line.replace(aal, "")
    line.replace(al, "")
    line.replace(am, "")
    line.replace(an, "")
    line.replace(ao, "")
    line.replace(ap, "")
    line.replace(aq, "")
    line.replace(aaq, "")
    line.replace(ar, "")
    line.replace(as_, "")
    line.replace(at, "")
    line.replace(au, "")
    line.replace(av, "")
    line.replace(aw, "")
    line.replace(ax, "")
    line.replace(ay, "")
    line.replace(az, "")
    line.replace(ba, "")
    line.replace(bb, "")
    line.replace(bd, "")
    line.replace(be, "")
    line.replace(bf, "")
    line.replace(bg, "")

    # new = ~ s / bh | bi / / g
    line.replace(bh, " ")
    line.replace(bi, " ")

    # new = ~ s /\r /\n / g
    line.replace("\r", "\n")
    # new = ~ s / ي / ی / g
    line.replace("ي", "ی")
    # new = ~ s / ك / ک / g
    line.replace("ك", "ک")

    line = line.replace("\-s\-", "_s_")
    # new = ~ s /\-s\- / _s_ / g
    # new =~ s/ـ+|ِ|ُ|َ|ْ|ٌ|ٍ//g
    # new = ~ s /﷼ / ریال / g
    line = line.replace("﷼", "ریال")
    # new = ~ s / ء | ء / ء / g
    # new = new.replace("", "")
    line = re.sub(r"ء|ء", "ء", line)
    # new = ~ s / ا | ا | ﺎ | ﺇ / ا / g
    line = re.sub(r"ﺇ|ﺎ|ا|ا", "ا", line)
    # new = ~ s / ب | ﺑ / ب / g
    line = re.sub(r"ب|ﺑ", "ب", line)
    # new = ~ s / ة | ۀ | ﮤ / ه / g
    line = re.sub(r"ة|ۀ|ﮤ", "ه", line)
    # new = ~ s / ت | ﺘ / ت / g
    line = re.sub(r"ت|ﺘ", "ت", line)
    # new = ~ s / ث / ث / g
    line = re.sub(r" ج|ﺟ", "ج", line)
    # new = ~ s / ج | ﺟ / ج / g
    line = re.sub(r"ج|ﺟ", "ج", line)
    # new = ~ s / ح / ح / g
    line = re.sub(r"ح", "ح", line)
    # new = ~ s / خ | ﺧ / خ / g
    line = re.sub(r"خ|ﺧ", "خ", line)
    # new = ~ s / د | ﺪ / د / g
    line = re.sub(r"د|ﺪ", "د", line)
    # new = ~ s / ذ / ذ / g
    line = re.sub(r"ذ", "ذ", line)
    # new = ~ s / ر | ﺮ / ر / g
    line = re.sub(r"ر|ﺮ", "ر", line)
    # new = ~ s / ز | ﺰ / ز / g
    line = re.sub(r"ز|ﺰ", "ز", line)
    # new = ~ s / س | ﺳ | ﺴ / س / g
    line = re.sub(r"س|ﺳ|ﺴ", "س", line)
    # new = ~ s / ش | ﺶ / ش / g
    line = re.sub(r" ش|ﺶ", "ش", line)
    # new = ~ s / ص | ﺻ / ص / g
    line = re.sub(r"ص|ﺻ", "ص", line)
    # new = ~ s / ض / ض / g
    line = re.sub(r"ض", "ض", line)
    # new = ~ s / ط | ﻃ / ط / g
    line = re.sub(r"ط|ﻃ", "ط", line)
    # new = ~ s / ظ / ظ / g
    line = re.sub(r"ظ", "ظ", line)
    # new = ~ s / ع / ع / g
    line = re.sub(r"ع", "ع", line)
    # line = ~ s / غ | ﻐ / غ / g
    line = re.sub(r"غ|ﻐ", "غ", line)
    # new = ~ s / ػ / ک / g
    line = re.sub(r"ػ", "ک", line)
    # new = ~ s / ؼ / ک / g
    line = re.sub(r"ؼ", "ک", line)
    # new = ~ s / ف | ﻔ / ف / g
    line = re.sub(r"ف|ﻔ", "ف", line)
    # new = ~ s / ق | ﻗ / ق / g
    line = re.sub(r"ق|ﻗ", "ق", line)
    # new = ~ s / ل / ل / g
    line = re.sub(r"ل", "ل", line)
    # new = ~ s / م / م / g
    line = re.sub(r"م", "م", line)
    # new = ~ s / ن / ن / g
    # new = re.sub(r"", "", new)
    # new = ~ s / ه / ه / g
    line = re.sub(r"ه", "ه", line)
    # new = ~ s / و / و / g
    line = re.sub(r"و", "و", line)
    # new = ~ s / ٯ / و / g
    line = re.sub(r"ٯ", "و", line)
    # new = ~ s / ٱ / ا / g
    line = re.sub(r"ٱ", "ا", line)
    # new = ~ s / ٲ / ا / g
    line = re.sub(r"ٲ", "ا", line)
    # new = ~ s / ٳ / ا / g
    line = re.sub(r"ٳ", "ا", line)
    # new = ~ s / ٴ // g
    line = re.sub(r"ٴ ", "", line)
    # new = ~ s / ٶ / ؤ / g
    line = re.sub(r"ٶ", "ؤ", line)
    # new = ~ s / ٸ / ئ / g
    line = re.sub(r"ٸ", "ئ", line)
    # new = ~ s / ٹ / ت / g
    line = re.sub(r"ٹ", "ت", line)
    # new = ~ s / ټ / ت / g
    line = re.sub(r"ټ", "ت", line)
    # new = ~ s / ٽ / ث / g
    line = re.sub(r"ٽ", "ث", line)
    # new = ~ s / پ / پ / g
    line = re.sub(r"پ", "پ", line)
    # new = ~ s / ٿ / ت / g
    line = re.sub(r"ٿ", "ت", line)
    # new = ~ s / ځ / ح / g
    line = re.sub(r"ځ", "ح", line)
    # new = ~ s / ڃ / ج / g
    line = re.sub(r"ڃ", "ج", line)
    # new = ~ s / ڄ / ج / g
    line = re.sub(r"ڄ", "ج", line)
    # new = ~ s / څ / خ / g
    line = re.sub(r"څ", "خ", line)
    # new = ~ s / چ / چ / g
    line = re.sub(r"چ", "چ", line)
    # new = ~ s / ڇ / چ / g
    line = re.sub(r"ڇ", "چ", line)
    # new = ~ s / ڈ / د / g
    line = re.sub(r"ڈ", "د", line)
    # new = ~ s / ډ / د / g
    line = re.sub(r"ډ", "د", line)
    # new = ~ s / ڌ / ذ / g
    line = re.sub(r"ڌ", "ذ", line)
    # new = ~ s / ڍ / د / g
    line = re.sub(r"ڍ", "د", line)
    # new = ~ s / ڑ / ر / g
    line = re.sub(r"ڑ", "ر", line)
    # new = ~ s / ڒ / ر / g
    line = re.sub(r"ڒ", "ر", line)
    # new = ~ s / ړ / ر / g
    line = re.sub(r"ړ", "ر", line)
    # new = ~ s / ڕ / ر / g
    line = re.sub(r"ڕ", "ر", line)
    # new = ~ s / ږ / ز / g
    line = re.sub(r"ږ", "ز", line)
    # new = ~ s / ڗ / ز / g
    line = re.sub(r"ڗ", "ز", line)
    # new = ~ s / ژ / ژ / g
    line = re.sub(r"ژ", "ژ", line)
    # new = ~ s / ښ / س / g
    line = re.sub(r"ښ", "س", line)
    # new = ~ s / ڞ / ض / g
    line = re.sub(r"ڞ", "ض", line)
    # new = ~ s / ڟ / ظ / g
    line = re.sub(r"ڟ", "ظ", line)
    # new = ~ s / ڡ / ف / g
    line = re.sub(r"ڡ", "ف", line)
    # new = ~ s / ڤ / ف / g
    line = re.sub(r"ڤ", "ف", line)
    # new = ~ s / ک / ک / g
    line = re.sub(r"ک", "ک", line)
    # new = ~ s / ڪ / ک / g
    line = re.sub(r"ڪ", "ک", line)
    # new = ~ s / ګ / ک / g
    line = re.sub(r"ګ", "ک", line)
    # new = ~ s / ڭ / ک / g
    line = re.sub(r"ڭ", "ک", line)
    # new = ~ s / گ / گ / g
    line = re.sub(r"گ", "گ", line)
    # new = ~ s / ڰ / گ / g
    line = re.sub(r"ڰ", "گ", line)
    # new = ~ s / ڱ / گ / g
    line = re.sub(r"ڱ", "گ", line)
    # new = ~ s / ڵ / ل / g
    line = re.sub(r"ڵ", "ل", line)
    # new = ~ s / ں / ن / g
    line = re.sub(r"ں", "ن", line)
    # new = ~ s / ڼ / ن / g
    line = re.sub(r"ڼ", "ن", line)
    # new = ~ s / ھ / ه / g
    line = re.sub(r"ھ", "ه", line)
    # new = ~ s / ۀ / ه / g
    line = re.sub(r"ۀ", "ه", line)
    # new = ~ s / ہ / ه / g
    line = re.sub(r"ہ", "ه", line)
    # new = ~ s / ۂ / ه / g
    line = re.sub(r"ۂ", "ه", line)
    # new = ~ s / ۃ / ه / g
    line = re.sub(r"ۃ", "ه", line)
    # new = ~ s / ۅ / و / g
    line = re.sub(r"ۅ", "و", line)
    # new = ~ s / ۆ / و / g
    line = re.sub(r"ۆ", "و", line)
    # new = ~ s / ۇ / و / g
    line = re.sub(r"ۇ", "و", line)
    # new = ~ s / ۈ / و / g
    line = re.sub(r"ۈ", "و", line)
    line = re.sub(r"ۉ", "و", line)
    line = re.sub(r"ۊ", "و", line)
    line = re.sub(r"ی", "ی", line)
    line = re.sub(r"ۍ", "ی", line)
    line = re.sub(r"ێ", "ی", line)
    line = re.sub(r"ۏ", "و", line)
    line = re.sub(r"ې|ﯼ|ﯽ|ﯾ", "ی", line)
    line = re.sub(r"ے", "ی", line)
    line = re.sub(r"ۓ", "ی", line)
    line = re.sub(r"ە", "ه", line)
    line = re.sub(r"ۥ", "و", line)
    line = re.sub(r"۽", "", line)
    line = re.sub(r"ݘ", "چ", line)
    line = re.sub(r"ݜ", "ش", line)
    line = re.sub(r"ݡ", "ف", line)
    line = re.sub(r"ݻ", "ی", line)
    line = re.sub(r"ﭖ", "پ", line)
    line = re.sub(r"ﭗ", "پ", line)
    line = re.sub(r"ﭘ | ﭘ", "پ", line)
    line = re.sub(r"ﭙ", "پ", line)
    line = re.sub(r"ﭞ", "ت", line)
    line = re.sub(r"ﭵ", "ج", line)
    line = re.sub(r"ﭺ", "چ", line)
    line = re.sub(r"ﭻ", "چ", line)
    line = re.sub(r"ﭼ|ﭼ", "چ", line)
    line = re.sub(r"ﭽ", "چ", line)
    line = re.sub(r"ﮊ", "ژ", line)
    line = re.sub(r"ﮋ", "ژ", line)
    line = re.sub(r"ﮎ|ﮐ|ﮑ|ﻛ", "ک", line)
    line = re.sub(r"ﮏ", "ک", line)
    line = re.sub(r"ﮐ", "ک", line)
    line = re.sub(r"ﮑ", "ک", line)
    line = re.sub(r"ﻙ|ﻚ", "ک", line)
    line = re.sub(r"ﮒ", "گ", line)
    line = re.sub(r"ﮓ|ﮓ", "گ", line)
    line = re.sub(r"ﮔ", "گ", line)
    line = re.sub(r"ﮕ|ﮕ", "گ", line)
    line = re.sub(r"ﮚ", "گ", line)
    line = re.sub(r"ﮤ|ۀ", "ه", line)
    line = re.sub(r"ﮥ", "ه", line)
    line = re.sub(r"ﮧ", "ه", line)
    line = re.sub(r"ﮩ", "ه", line)
    line = re.sub(r"ﮫ", "ه", line)
    line = re.sub(r"ﮬ", "ه", line)
    line = re.sub(r"ﮭ", "ه", line)
    line = re.sub(r"ﮮ", "ی", line)
    line = re.sub(r"ﮯ", "ی", line)
    line = re.sub(r"ﯚ", "و", line)
    line = re.sub(r"ﯼ", "ی", line)
    line = re.sub(r"ﯽ", "ی", line)
    line = re.sub(r"ﯾ", "ی", line)
    line = re.sub(r"ﯿ", "ی", line)
    line = re.sub(r"ﳊ", "لح", line)
    line = re.sub(r"ﷲ", "الله", line)
    line = re.sub(r"ﺀ", "ء", line)
    line = re.sub(r"ﺁ", "آ", line)
    line = re.sub(r"ﺂ", "آ", line)
    line = re.sub(r"ﺃ", "أ", line)
    line = re.sub(r"ﺄ", "أ", line)
    line = re.sub(r"ﺅ", "ؤ", line)
    line = re.sub(r"ﺆ", "ؤ", line)
    line = re.sub(r"ﺈ", "ا", line)
    line = re.sub(r"ﺉ", "ئ", line)
    line = re.sub(r"ﺊ", "ئ", line)
    line = re.sub(r"ﺋ", "ئ", line)
    line = re.sub(r"ﺌ", "ئ", line)
    line = re.sub(r"ﺍ", "ا", line)
    line = re.sub(r"ﺎ", "ا", line)
    line = re.sub(r"ﺏ", "ب", line)
    line = re.sub(r"ﺐ", "ب", line)
    line = re.sub(r"ﺑ", "ب", line)
    line = re.sub(r"ﺒ", "ب", line)
    line = re.sub(r"ﺓ", "ه", line)
    line = re.sub(r"ﺔ", "ه", line)
    line = re.sub(r"ﺕ", "ت", line)
    line = re.sub(r"ﺖ", "ت", line)
    line = re.sub(r"ﺗ", "ت", line)
    line = re.sub(r"ﺘ", "ت", line)
    line = re.sub(r"ﺙ", "ث", line)
    line = re.sub(r"ﺚ", "ث", line)
    line = re.sub(r"ﺛ", "ث", line)
    line = re.sub(r"ﺜ", "ث", line)
    line = re.sub(r"ﺝ", "ج", line)
    line = re.sub(r"ﺞ", "ج", line)
    line = re.sub(r"ﺟ", "ج", line)
    line = re.sub(r"ﺠ", "ج", line)
    line = re.sub(r"ﺡ", "ح", line)
    line = re.sub(r"ﺢ", "ح", line)
    line = re.sub(r"ﺣ", "ح", line)
    line = re.sub(r"ﺤ", "ح", line)
    line = re.sub(r"ﺥ", "خ", line)
    line = re.sub(r"ﺦ", "خ", line)
    line = re.sub(r"ﺧ", "خ", line)
    line = re.sub(r"ﺨ", "خ", line)
    line = re.sub(r"ﺩ", "د", line)
    line = re.sub(r"ﺪ", "د", line)
    line = re.sub(r"ﺫ", "ذ", line)
    line = re.sub(r"ﺬ", "ذ", line)
    line = re.sub(r"ﺭ", "ر", line)
    line = re.sub(r"ﺮ", "ر", line)
    line = re.sub(r"ﺯ", "ز", line)
    line = re.sub(r"ﺰ", "ز", line)
    line = re.sub(r"ﺱ", "س", line)
    line = re.sub(r"ﺲ", "س", line)
    line = re.sub(r"ﺳ", "س", line)
    line = re.sub(r"ﺴ", "س", line)
    line = re.sub(r"ﺵ", "ش", line)
    line = re.sub(r"ﺶ", "ش", line)
    line = re.sub(r"ﺷ", "ش", line)
    line = re.sub(r"ﺸ", "ش", line)
    line = re.sub(r"ﺹ", "ص", line)
    line = re.sub(r"ﺺ", "ص", line)
    line = re.sub(r"ﺻ", "ص", line)
    line = re.sub(r"ﺼ", "ص", line)
    line = re.sub(r"ﺽ", "ض", line)
    line = re.sub(r"ﺾ", "ض", line)
    line = re.sub(r"ﺿ", "ض", line)
    line = re.sub(r"ﻀ", "ض", line)
    line = re.sub(r"ﻁ", "ط", line)
    line = re.sub(r"ﻂ", "ط", line)
    line = re.sub(r"ﻃ", "ط", line)
    line = re.sub(r"ﻄ", "ط", line)
    line = re.sub(r"ﻅ", "ظ", line)
    line = re.sub(r"ﻆ", "ظ", line)
    line = re.sub(r"ﻇ", "ظ", line)
    line = re.sub(r"ﻈ", "ظ", line)
    line = re.sub(r"ﻉ", "ع", line)
    line = re.sub(r"ﻊ", "ع", line)
    line = re.sub(r"ﻋ", "ع", line)
    line = re.sub(r"ﻌ", "ع", line)
    line = re.sub(r"ﻍ", "غ", line)
    line = re.sub(r"ﻎ", "غ", line)
    line = re.sub(r"ﻏ", "غ", line)
    line = re.sub(r"ﻐ", "غ", line)
    line = re.sub(r"ﻑ", "ف", line)
    line = re.sub(r"ﻒ", "ف", line)
    line = re.sub(r"ﻓ", "ف", line)
    line = re.sub(r"ﻔ", "ف", line)
    line = re.sub(r"ﻕ", "ق", line)
    line = re.sub(r"ﻖ", "ق", line)
    line = re.sub(r"ﻗ", "ق", line)
    line = re.sub(r"ﻘ", "ق", line)
    line = re.sub(r"ﻙ", "ک", line)
    line = re.sub(r"ﻚ", "ک", line)
    line = re.sub(r"ﻛ", "ک", line)
    line = re.sub(r"ﻜ", "ک", line)
    line = re.sub(r"ﻝ", "ل", line)
    line = re.sub(r"ﻞ", "ل", line)
    line = re.sub(r"ﻟ", " ل", line)
    line = re.sub(r" ﻠ|ﻟ|ﻠ", " ل", line)
    line = re.sub(r"ﻡ", "م", line)

    line = re.sub(r"ﻡ", "م", line)
    line = re.sub(r"ﻢ", "م", line)
    line = re.sub(r"ﻣ", "م", line)
    line = re.sub(r"ﻤ|ﻣ|ﻤ", "م", line)
    line = re.sub(r"ﻥ|ﻦ|ﻧ|ﻨ", "ن", line)
    line = re.sub(r"ﻦ", "ن", line)
    line = re.sub(r"ﻧ", "ن", line)
    line = re.sub(r"ﻨ", "ن", line)
    line = re.sub(r"ﻩ", "ه", line)
    line = re.sub(r"ﻪ|ﻪ", "ه", line)
    line = re.sub(r"ﻫ", "ه", line)
    line = re.sub(r"ﻬ", "ه", line)
    line = re.sub(r"ﻭ", "و", line)
    line = re.sub(r"ﻮ|ﻮ", "و", line)
    line = re.sub(r"ﻰ", "ی", line)
    line = re.sub(r"ﻱ", "ی", line)
    line = re.sub(r"ﻲ", "ی", line)
    line = re.sub(r"ﻳ", "ی", line)
    line = re.sub(r"ﻴ|ﻴ", "ی", line)
    line = re.sub(r"ﻵ", "لا", line)
    line = re.sub(r"ﻷ", "لا", line)
    line = re.sub(r"ﻸ", "لا", line)
    line = re.sub(r"ﻻ", "لا", line)
    line = re.sub(r"ﻼ", "لا", line)
    line = re.sub(r"ﻶ", "لا", line)
    line = re.sub(r"ﻹ", "لا", line)

    line = re.sub(r"０|۰|٠", "0", line)
    line = re.sub(r"１|۱|١", "1", line)
    line = re.sub(r"２|۲|٢", "2", line)
    line = re.sub(r"３|۳|٣", "3", line)
    line = re.sub(r"４|۴|٤", "4", line)
    line = re.sub(r"５|۵|٥", "5", line)
    line = re.sub(r"６|۶|٦", "6", line)
    line = re.sub(r"７|۷|٧", "7", line)
    line = re.sub(r"８|۸|٨", "8", line)
    line = re.sub(r"９|۹|٩", "9", line)




    line = re.sub(r"؈", "و", line)
    line = re.sub(r"ܘ", "و", line)
    line = re.sub(r"‘", "`", line)
    line = re.sub(r"’", "'", line)
    line = re.sub(r"‚", ",", line)
    line = re.sub(r"“", '"', line)
    line = re.sub(r"”", '"', line)

    line = re.sub(r"آآ+", "آ", line)
    line = re.sub(r"ااا+", "ا", line)
    line = re.sub(r"ببب+", "بب", line)
    line = re.sub(r"پپپ+", "پ", line)
    line = re.sub(r"تتت+", "ت", line)
    line = re.sub(r"ثثث+", "ث", line)
    line = re.sub(r"ججج+", "ج", line)
    line = re.sub(r"چچچ+", "چ", line)
    line = re.sub(r"ححح+", "ح", line)
    line = re.sub(r"خخخ+", "خ", line)
    line = re.sub(r"ددد+", "د", line)
    line = re.sub(r"ذذذ+", "ذ", line)
    line = re.sub(r"ررر+", "ر", line)
    line = re.sub(r"ززز+", "ز", line)
    line = re.sub(r"ژژژ+", "ژ", line)
    line = re.sub(r"سسس+", "س", line)
    line = re.sub(r"ششش+", "ش", line)
    line = re.sub(r"صصص+", "ص", line)
    line = re.sub(r"ضضض+", "ض", line)
    line = re.sub(r"ططط+", "ط", line)
    line = re.sub(r"ظظظ+", "ظ", line)
    line = re.sub(r"ععع+", "ع", line)
    line = re.sub(r"غغغ+", "غ", line)
    line = re.sub(r"ففف+", "ف", line)
    line = re.sub(r"ققق+", "ق", line)
    line = re.sub(r"ککک+", "ک", line)
    line = re.sub(r"گگگ+", "گ", line)
    line = re.sub(r"للل+", "ل", line)
    line = re.sub(r"ممم+", "م", line)
    line = re.sub(r"ننن+", "ن", line)
    line = re.sub(r"ووو+", "و", line)
    line = re.sub(r"ههه+", "ه", line)
    line = re.sub(r"ییی+|ییی", "یی", line)
    line = re.sub(r"أأأ+", "أ", line)
    line = re.sub(r"إإإ+", "إ", line)
    line = re.sub(r"ؤؤؤ+", "ؤ", line)
    line = re.sub(r"ئئئ+", "ئ", line)
    line = re.sub(r"و+|و_s_", "و", line)
    line = re.sub(r"ـ", "-", line)
    """ WHAT are these"""
    line = re.sub(r"ً+", "ً", line)
    line = re.sub(r"_+", "_", line)
    line = re.sub(r"\-+", r"\-", line)
    line = re.sub(r"\*+", r"\*", line)
    line = re.sub(r"\(+", "\(", line)
    line = re.sub(r"  \)+", "\)", line)
    line = re.sub(r"\++", "\+", line)
    line = re.sub(r"\=+", "=", line)
    line = re.sub(r"\!+", "\!", line)
    line = re.sub(r"\?+", "\?", line)
    line = re.sub(r"‘", "'", line)
    line = re.sub(r"’", "'", line)
    line = re.sub(r"‚", ",", line)
    line = re.sub(r"“", '"', line)
    line = re.sub(r"”", '"', line)
    line = re.sub(r"ً+", "ً", line)
    # line = re.sub(r"([0 - 9])_s_", "$1", line)
    p = re.compile("(.*)([0-9])_s_(.*)")
    # print(line)
    result = p.search(line)
    if result is not None:
        line = result.group(1) + result.group(2) + " " + result.group(3)

    re.sub(r"\r", "\n", line)
    """WHat is this """

    # line = re.sub(r"* <* ([A - Z])", r"\n<$1", line) , commented in perl file too

    """WHat is this """

    # line = re.sub(r"* >", " > \n", line)

    line = re.sub(r"ییَ", " ییَ", line)
    # line = re.sub(r"s_", " ", line)
    line = re.sub(r"آ_s_", "آ", line)
    line = re.sub(r"ا_s_", "ا", line)
    line = re.sub(r"_s_ ", " ", line)
    line = re.sub(r" _s_", " ", line)

    line = re.sub(r"0", "\.", line)
    line = re.sub(r"0\$", "\.", line)

    # line = re.sub(r"(\d),(\d)", r"$1$2", line)
    p = re.compile("(.*)(\d),(\d)(.*)")
    result = p.search(line)
    if result is not None:
        line = result.group(1) + result.group(2) + " " + result.group(3) + result.group(4)

    # line = re.sub(r"(\d)،(\d)", r"$1$2", line)
    p = re.compile("(.*)(\d)،(\d)(.*)")
    result = p.search(line)
    if result is not None:
        line = result.group(1) + result.group(2) + " " + result.group(3) + result.group(4)



    # line = re.sub(r"(\d)ر(\d)", r"$1$2", line)
    p = re.compile("(.*)(\d)ر(\d)(.*)")
    result = p.search(line)
    if result is not None:
        line = result.group(1) + result.group(2) + " " + result.group(3) + result.group(4)

    #	$new =~ s/(\D)(\d)/$1 $2/g
    #	$new =~ s/(\d)(\D)/$1 $2/g
    """ was commented on perl code but it seemed necessary"""
    p = re.compile("(.*)(\d)(\D)(.*)")
    result = p.search(line)
    if result is not None:
        line = result.group(1) + result.group(2) + " " + result.group(3) + result.group(4)


    p = re.compile("(.*)(\D)(\d)(.*)")
    result = p.search(line)
    if result is not None:
        line = result.group(1) + result.group(2) + " " + result.group(3) + result.group(4)
    """----------------------------------------------------------------"""


    p = re.compile("(.*)(\d)_s_(\D)(.*)")
    result = p.search(line)
    if result is not None:
        line = result.group(1)  + result.group(2) + " " + result.group(3)  + result.group(4)
    # line = re.sub(r"(\d)_s_(\D)", "$1 $2", line)
    """WHat is this """
    # line = re.sub(r"+", " ", line)
    #	$new =~ s/<N ([A-Z])([A-Z]) /<N $1$2/g , commented on perl too

    line = re.sub(r"\-َ ", "\-", line)
    line = re.sub(r"-", " - ", line)

    line = re.sub(r"\.\.\.\.\.\.\.", "…", line)
    line = re.sub(r"\۰ *  \۰ *  \۰", "…", line)

    line = re.sub(r"\.\.\.\.", "…", line)
    line = re.sub(r"\۰\۰\۰\۰", "…", line)

    line = re.sub(r"\.\.\.", "…", line)
    line = re.sub(r"\۰\۰\۰ ", "…", line)

    line = re.sub(r"ا\.\.\.|ا_s_\.\.\.", "ا…", line)
    line = re.sub(r"ا\.\.|ا_s_\.\.", "ا…", line)

    line = re.sub(r"/¬", "_s_", line)
    # $quot = "'"
    line = re.sub(r"\t", " ", line)

    line = re.sub(r"||||||||||||||||||||||||||||||­|ă|Ă|Ǻ|ȧ|Ȧ|Ǡ|Ą|ǣ|Ɓ|Ċ|ƈ|ǆ|ė|ǵ|Ĝ|ģ|Ģ|ƣ|Ħ|Ĭ|ĩ|į|Ȉ|Ǩ|Ƙ|Ľ|ǈ|ǹ|Ǹ|ŋ|ǋ|ǒ|Ɵ|ő|ȯ|Ǫ|ȍ|Ơ|ŕ|Ř|Ŗ|Ś|ſ|Ť|ŧ|ţ|Ţ|Ŭ|ǔ|Ů|ű|Ű|ų|Ų|Ū|ȕ|ư|Ư|ŵ|Ŵ|ƿ|Ŷ|Ƴ|ź|Ź|ż|Ż|ƶ|Ƶ|Ȥ|Ʒ|Ǯ|ϫ|Ϫ|Ϩ|ϩ|ϳ|ϰ|Ϧ|ϧ|ϭ|Ϯ|ϯ|Ύ|Ϋ|Ϥ|ϥ|ख़|घ|બ|ద|ণ|ম|ං|ඟ|♥|×|❀|◕|‿|㋡|ღ|¤|~|\[ +|●|\|³|§+|¡|¢|£|¤|¥|¦|§|¨|©|ª|¬|­|®|¯|°|±|²|³|´|µ|¶|·|¸|¹|º|¼|½|¾|¿|À|Á|Â|Ã|Ä|Å|Æ|Ç|È|É|Ê|Ë|Ì|Í|Î|Ï|Ð|Ñ|Ò|Ó|Ô|Õ|Ö|×|Ø|Ù|Ú|Û|Ü|Ý|Þ|ß|à|á|â|ã|ä|å|æ|ç|è|é|ê|ë|ì|í|î|ï|ð|ñ|ò|ó|ô|õ|ö|ø|ù|ú|û|ü|ý|þ|ÿ|Ā|ā|Ć|ć|ĉ|ċ|Č|č|ď|Đ|đ|ē|ĕ|ę|Ě|ě|ĝ|Ğ|ğ|ġ|Ĥ|ĥ|Ĩ|Ī|ī|İ|ı|Ĵ|ĵ|ķ|ĸ|Ĺ|ļ|ľ|ł|ń|Ņ|ň|Ŋ|Ō|ō|Œ|œ|Ŕ|ŗ|ř|ś|Ŝ|ŝ|Ş|ş|Š|š|ť|Ŧ|Ũ|ū|ŭ|ů|Ÿ|Ž|ž|ƀ|ƃ|Ǝ|Ə|ƒ|Ɠ|ƛ|Ɲ|ƞ|ơ|Ƨ|Ʊ|ƴ|Ƹ|ǁ|ǅ|Ǎ|ǎ|ǐ|Ǖ|ǘ|ǝ|ǟ|ǡ|ǥ|Ǧ|ǻ|ǽ|ǿ|ȁ|Ȍ|Ȑ|Ȗ|Ș|ș|ȡ|ȥ|Ȩ|ȩ|Ȫ|ȴ|Ⱥ|Ɋ|ɐ|ɑ|ɒ|ɔ|ɕ|ə|ɟ|ɢ|ɥ|ɨ|ɪ|ɭ|ɯ|ɱ|ɸ|ɹ|ʁ|ʅ|ʊ|ʋ|ʌ|ʍ|ʐ|ʒ|ʗ|ʘ|ʞ|ʡ|ʤ|ʥ|ʲ|ʵ|ʺ|ʻ|ʼ|ʽ|ʾ|ʿ|˅|ˆ|ˇ|ˈ|ˉ|ˊ|ˌ|ˏ|˓|˔|˙|˚|˛|˜|˝|˟|˥|˨|˵|˶|˿|̀|́|̂|̃|̄|̅|̈|̋|̎|̏|̓|̕|̗|̜|̟|̠|̦|̨|̪|̫|̮|̯|̱|̲|̷|̾|̈́|͕|͘|͜|͡|ͫ|;|΁|΃|΄|΅|·|Ί|Ώ|Α|Β|Γ|Δ|Ε|Ζ|Η|Θ|Ι|Κ|Λ|Μ|Ν|Ξ|Ο|Π|Ρ|Σ|Τ|Υ|Φ|Χ|Ψ|Ω|ά|έ|ή|ί|ΰ|α|β|γ|δ|ε|ζ|η|θ|ι|κ|λ|μ|ν|ξ|ο|π|ρ|ς|σ|τ|υ|φ|χ|ψ|ω|ϊ|ό|ύ|ϐ|ϖ|Ϙ|ϙ|ϛ|Ϝ|ϝ|Ϡ|ϡ|ϣ|ϲ|Ϻ|ϻ|Ͼ|Ͽ|Є|І|Ї|Ј|Љ|Џ|А|Б|В|Г|Д|Е|Ж|З|И|Й|К|Л|М|Н|О|П|Р|С|Т|У|Ф|Х|Ц|Ч|Ш|Ь|Э|Ю|Я|а|б|в|г|д|е|ж|з|и|й|к|л|м|н|о|п|р|с|т|у|ф|х|ц|ч|ш|щ|ъ|ы|ь|э|ю|я|ѐ|ё|ђ|є|ѕ|і|ї|ј|ѡ|ѳ|ѷ|҂|҉|ҍ|Ґ|ґ|җ|Ҙ|ҡ|Ң|ҥ|Ұ|ҳ|Ҷ|Ҹ|Һ|ҽ|Ӂ|ӑ|Ә|ә|ӛ|ӡ|ӥ|Ө|ӷ|ӻ|Ԁ|Ԑ|Ԙ|ԡ|Ԩ|Ԫ|ԫ|԰|Ա|Գ|Ը|Թ|Ի|Կ|Հ|Յ|Պ|Վ|Ր|Փ|՚|՞|ա|գ|դ|ե|զ|է|ը|թ|ի|լ|ծ|կ|հ|ղ|ճ|մ|յ|ն|շ|ո|չ|պ|ռ|ս|վ|տ|ր|ց|ւ|փ|ք|֬|֮|ְ|ֱ|ֲ|ִ|ֵ|ֶ|ַ|ָ|ֹ|ֺ|ּ|ֽ|ֿ|ׁ|׃|א|ב|ג|ד|ה|ו|ז|ח|ט|י|ך|כ|ל|ם|מ|ן|נ|ס|ע|ף|פ|ץ|צ|ק|ר|ש|ת|׳|״|׷|؀|؄|؆|؊|؍|؎|؛|؜|ؠ|ـ|٪|٫|٬|٭|ٮ|۔|۝|۞|۩|܀|ܐ|ܒ|ܓ|ܗ|ܝ|ܣ|ܨ|ܪ|ܮ|ܽ|ސ|ޘ|ޯ|ߤ|߯|߲|߾|ࠀ|࠽|࡮|ࡰ|ࡴ|࣑|ऀ|ं|अ|इ|ई|उ|ए|ओ|क|ख|ग|च|छ|ज|ञ|ड|ण|त|थ|द|ध|न|प|फ|म|य|र|ल|व|श|ष|स|ह|़|ा|ि|ी|ु|ू|े|ै|्|।|०|१|२|३|५|७|८|९|ল|ৄ|ড়|ঢ়|ਰ|ਾ|ੈ|૛|ୌ|୬|୶|ஐ|க|ஜ|ட|த|஦|ப|ம|ல|ள|ழ|வ|ி|ீ|ு|ை|்|ఀ|ః|ఈ|ಌ|ല|ോ|ධ|න|඲|ඳ|ප|ඵ|බ|භ|ම|ඹ|฀|ก|ค|ง|ช|ซ|ญ|ฐ|ฑ|ณ|ท|น|ป|พ|ฟ|ม|ร|ฤ|ษ|ห|า|ุ|เ|ํ|๏|๐|๑|ຨ|་|༳|ཏ|བ|མ|འ|ལ|ཱ|ི|ྭ|ླ|࿠|က|ဈ|ဠ|ေ|ဴ|၅|၇|၈|ႆ|჌|ღ|ᄋ|ᆪ|ኧ|ᙐ|ᝠ|ᮈ|ᴉ|ᵐ|ḙ|ḣ|ḥ|ṁ|ṣ|ṭ|ṿ|ẖ|ạ|ả|ặ|ể|ệ|ộ|Ἀ|Ἐ|Ἓ|Ἰ|ὅ|Ὀ|ὠ|ὶ|ᾍ|ᾫ|ᾱ|ῆ|Ὲ|ῠ|ῦ|ῶ|Ὸ| | | | | | | | | | |​|‌|‍|‐|‑|‒|–|—|―|‗|‘|’|‚|‛|“|”|„|‟|†|‡|•|․| |′|″|‹|›|‼|‿|⁄|⁪|⁫|⁬|⁭|⁮|⁯|⁴|⁵|⁺|⁻|⁽|⁾|ⁿ|₃|₌|₡|₣|₤|₧|₪|€|₰|⃝|ℓ|™|Ω|ℸ|⅓|⅔|←|↑|→|↓|↔|↰|↳|⇆|∆|∇|∈|∑|−|∕|∗|∙|∞|≈|≠|≡|≤|≥|⊙|⋘|⋙|⌒|⌡|⌢|⌣|⌬|⍍|⍒|⍔|⍯|⍰|⎡|⎦|⎨|⎪|⎭|⏀|⏾|ⓐ|ⓓ|ⓕ|ⓢ|─|│|┊|┌|┏|┐|┓|└|┘|├|┤|┬|┴|┼|═|║|╒|╓|╔|╕|╖|╗|╘|╙|╚|╛|╜|╝|╞|╟|╡|╢|╣|╤|╥|╦|╧|╨|╩|╪|╫|╬|▀|▂|▄|█|▌|░|▒|▓|■|□|▦|▧|▩|▪|▫|▬|▲|►|▼|◀|◄|◈|◊|○|◌|●|◕|◘|◙|◠|◡|◦|☀|☁|☂|★|☆|☠|☪|☹|☺|☻|☼|♀|♂|♠|♡|♣|♥|♦|♩|♪|♫|♬|♭|⛐|✀|✂|✍|✎|✓|✔|✖|✗|✘|✟|✦|✧|✪|✫|✮|✰|✲|✴|✸|✿|❀|❈|❉|❒|❤|❥|⭠|ⱀ|ⴸ|⻨|⼀|　|、|。|《|》|「|」|『|』|【|】|ぁ|あ|ぃ|い|う|え|お|か|が|き|く|け|こ|さ|し|じ|す|せ|そ|た|だ|っ|て|で|と|ど|な|に|ね|の|は|へ|ま|み|め|も|や|よ|ら|り|る|れ|ろ|わ|を|ん|ア|イ|エ|オ|カ|ガ|キ|ギ|ク|グ|ゴ|サ|シ|ス|ダ|ッ|ツ|デ|ト|ニ|ヌ|バ|ビ|フ|ブ|ボ|マ|ミ|ム|メ|ヤ|ユ|ラ|リ|ル|ロ|ン|・|ー|ㄱ|ㄲ|ㄸ|ㅁ|ㅃ|ㅆ|ㅇ|ㅉ|ㅏ|㇘|㋡|㙐|㜀|㷐|㹀|㺗|㻋|㾡|㿨|䅉|䐝|䐻|䑸|䘯|䙃|䙆|䞘|䤰|䪰|一|丁|七|上|下|不|与|世|両|中|为|主|义|之|乔|也|习|书|了|争|事|二|于|亐|亚|交|产|人|什|仁|介|从|他|仠|以|们|仰|件|任|份|伊|休|会|传|伦|伯|但|位|低|住|何|佛|作|你|使|供|依|侯|俗|保|信|倈|們|倫|假|偉|做|側|偽|傘|優|儿|元|先|光|克|免|內|全|八|六|兰|关|其|典|兹|内|冒|写|冠|准|出|切|列|刘|利|制|則|創|功|加|劣|助|勇|勒|動|務|勢|化|区|十|千|升|单|南|卜|占|卡|历|厚|原|去|参|及|友|双|反|发|取|古|台|史|叶|吃|合|名|后|吠|否|员|周|命|和|哈|响|哲|商|喋|喜|嗎|嘱|四|回|因|园|围|国|图|國|圣|在|地|坂|坦|埃|城|基|報|場|塑|塔|塸|士|壽|备|多|夜|夠|大|天|太|夫|头|奇|套|女|好|如|妇|姆|委|姿|威|孈|子|孔|字|孟|学|孩|它|守|宋|宗|定|宣|家|密|實|对|寻|导|対|對|少|尔|尘|尼|展|山|峰|州|工|差|己|已|巴|布|帆|希|帝|干|平|年|广|庄|庆|府|度|座|庭|廡|建|开|式|引|张|录|影|彼|得|德|心|必|忽|怈|思|性|怪|总|恒|息|恺|情|意|愛|態|慕|戈|戏|成|我|或|戦|所|手|扎|打|扠|批|把|抵|拉|拋|拦|持|指|挥|据|授|採|探|描|提|援|摘|摩|撃|播|撮|撰|擊|擠|攸|攻|政|故|敗|教|文|斯|新|方|施|无|日|早|时|明|易|昡|是|晁|晩|普|智|最|會|月|有|朗|未|本|杀|条|来|杨|松|林|果|某|柳|栏|株|核|根|格|桑|梅|棒|棚|椅|極|概|榨|榮|橋|機|歌|正|此|武|段|殺|母|每|比|气|水|永|求|汉|江|決|法|泘|波|津|派|济|海|消|涛|清|渠|渤|游|湧|湾|滝|演|漫|漱|澄|濕|灑|火|為|烈|热|然|爾|版|牛|牧|物|犬|献|獘|玉|王|玛|现|珠|理|琈|瑈|瑪|瑴|生|用|田|由|申|电|界|當|白|百|的|盖|監|目|看|真|眼|着|睛|督|睿|知|础|碐|磨|社|祈|祝|神|祭|祷|福|私|种|科|称|程|稱|穆|站|章|童|第|答|简|箴|米|系|絶|經|網|緰|練|繁|约|纯|纲|组|终|经|给|统|继|网|罕|羅|美|羽|老|者|而|耶|聖|肉|肚|肯|胜|能|脅|脆|腥|自|致|與|舍|舒|色|艾|节|花|英|茂|莖|菲|萨|蒰|蔡|虹|蚈|血|衡|衣|袖|被|褐|襲|西|要|見|親|解|言|訓|記|設|許|訳|評|試|詩|話|認|說|議|训|讲|论|设|词|译|诗|话|语|说|谁|谈|谓|谟|谷|賢|贤|贾|赛|赫|超|越|足|路|跳|車|軐|轨|转|载|辞|辩|达|过|迪|追|退|逈|选|逊|這|通|造|週|逼|道|違|遗|那|郎|部|郵|都|醒|里|重|鉴|錢|錱|鎐|锦|长|門|間|问|间|阿|陀|际|降|陰|際|雄|集|雖|雷|需|霍|革|鞀|鞋|領|领|题|风|館|馆|首|験|驈|马|高|魚|鮘|鰰|鲁|鳴|麼|黐|默|鼓|鼯|鼻|齐|ꊠ|ꌀ|ꓰ|Ꙕ|ꞌ|ꢰ|가|것|공|국|기|김|까|나|난|남|넷|누|는|닀|니|다|딐|따|때|땐|떐|라|먀|먹|미|박|밤|방|보|복|쀐|성|소|손|송|숙|시|신|아|양|여|연|영|오|올|용|우|원|유|윤|은|을|의|이|인|일|자|장|정|조|종|주|지|진|챀|채|척|쳐|최|펐|퓒|한|할|합|해|행|현|혜|혡|환|훿|희|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||廊|麗|逸|難|滛|齃|ﬁ|ﬤ|טּ|לּ|ﰸ|ﱠ|ﱢ|﴾|﴿|ﷺ|﷼|ﹾ|﻿|＼|｜|｡|･|￡|￦|￹|�|📖|🔴|🔵|😁|😄|😌|😏|😜|😝|😞|😡|😲|😳|🙌|󰀡|􀀯|􀀶|􀀹|􀁚|􀂃|􀂊|􀂌|􀂏|􀂙|􀂽|􀂾|􀆡| |ٔ|\ ^ +|\=|‍|¨|°|µ|┬|¯|­|¡|¿|¸|§|©|®|™|¤|¢|$|£|¥|₩|€|±|━|←|→|↑|↓|٭|ْ|💫|✍|¼|¾|Á|à|À|â|Â|å|Å|ä|Ä|ã|Ã|ā|ȁ|Ȃ|ª|æ|Æ|Ć|Č|ç|Ç|ď|ð|Ð|é|É|è|È|ê|Ê|ë|Ë|ę|ɛ|ğ|ģ|ħ|í|Í|ì|Ì|î|Î|ï|Ï|ı|İ|ĵ|ķ|Ł|ñ|Ñ|ó|Ó|ò|Ò|ô|Ô|ö|Ö|õ|Õ|ø|Ø|º|ŕ|ś|š|ş|Ş|ß|ú|ù|Ù|û|ü|Ü|ý|ÿ|þ|Þ|α|ε|ϩ|ι|λ|Ξ|ο|Π|σ|Ψ|ψ|Ω|а|б|в|г|Г|д|Ԁ|Ѓ|ђ|Ђ|е|є|Є|Ё|з|и|й|к|ҟ|м|н|Н|Ԋ|о|п|П|р|с|С|т|у|х|щ|ы|ь|э|ю|་|Հ|օ|ֆ|ღ|अ|आ|ऐ|औ|क|ख़|ग|च|झ|ट|ठ|ड|ढ|त|न|प|ब|भ|म|य|र|ल|स|ह|ा|ी|ु|े|ो|ਆ|ਃ|ം|ႁ| |一|乐|你|呵|嘎|噡|囧|土|夳|她|寷|峺|年|後|快|慢|慤|慭|慲|慳|扥|扩|护|挀|搀|摡|摥|摩|攠|敢|敤|敩|新|方|昀|明|来|栀|桡|桩|椠|楫|楮|歨|氠|污|法|洠|浡|浩|浮|渠|湥|漠|潬|潯|濚|灯|無|牡|猀|獥|琀|瑡|瑯|癡|的|硃|磙|祡|祤|祥|穩|縀|縆|缀|翻|肘|胟|蘆|蠀|说|过||✍|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||ݦ|਀|⨀|ส|วั|ส|ดี|ี|ใ|่|ྉ|💊|💉|🏡|ჿ|ᄀ̭|ᾨ|῕|㰀|※|⁥|⁰|₹|⃣|ۚ|ᾨ|℃|∠|∨|〒|≯|∨|↗|↙|↻|⇣|√|⇦|≦|≧|⏰|⏱|ⓖ|ⓘ|ⓚ|ⓨ|ⓞ|ⓤ|┛|╭|∩|╮|﹌|╭|╯|▽|｀|゜|◆|◇|◎|☄|☔️|☕️|☘|🍁|🍃|☝|🏻|️|🏻️|☞|☫|♊️|♊|☯|♏️|♏|♧|♻️|♻|🛡|⚔|⚘|⚜|⚡|⚫️|⚫|⛄|🌞|🏰|🏭|🏨|🏠|🏆|🐎|🎱|🏈|🏁|🎳|⚽|🚘|🚗|🚦|💒|✅|䘀|✆|䠀|✈️|✈|✉|📨|📩|📲|📱|🎍|✊|🏻|✋|✌|👋|👏|🌎|🌍|😉|💚|🏼|👌|🏽|💝|😐|😘|✨|✳|✳️|✳|✹|✹|❄|❇|⭐|❣|➖|➻|⠆|⨆|⼆|䔆|ㄆ|䌀|䠆|䘆|䨀|꼆|䜀|䨀|㌀|⤀|㤀|㤆|Ⰰ|䜆|㈆|㌆|䐀|⬅️|⬅|↕|⬇|➡|⭕️|⭕|➕|Ⲵ|鿻|⴩|〰|ご|ざ|㐆|ㄒ|ㄏ|ㅈ|䐆|䔀|㌵|㘲|㘵|㜶|㔀|㘰|㠀ଂ|㠅|䨆|䌆|㤵|⤍|䄆|㜆|㄀|㔆|ꭐ|〆|럻|ᒯ|렀|많|받|으|세|요|새|췿|폡|헿|샥|||||||||❌|🐒|🏬|🌄|🚃|🏤|🏪|🗾|💓|👍|＾|ｉ|～|￣|＊|！|﹏|🙊|😭|💯|￧|||||๧||೏|＀|ᦟ|ᄻ|＇|ﰝ|￫|᱿||🀼󞐼󞐮|🀼󞐼|🆔|🆗|🇮🇷|🇰🇷|🇨🇦|🇹🇯|🇹|🇰|🇴|🇺🇸|🇺🇬|🇺🇦|🇱🇨|🇸🇿|🇻🇪|🇻🇳|🇼|🇸🇷|🇿|🇷|🇫|🇺🇸|🇲🇸|🇩|🇻🇺|🇳|🇸|🇭|🇱|🇪|🇺|🌀|🌅|🌈|🌻|🌉|🌊|🌐|🌕|🌙|🌛|🌜|🔑|🌟|💎|💥|🌧|🌨|🌱|🌵|🌲|🌳|🌴|🌷|🌸|🌹|🌺|🖐|🌼|😚|🎄|🌾|🌿|🍀|🍂|💞|🐡|🙇|🎼|🎶|🍄|🍅|🌽|🍌|🍊|🍏|🍯|🍇|🍒|🍐|🍍|🍉|🍆|🍠|🍓|🍭|🍪|🍫|🍋|🍎|🍑|🍕|😿|🍢|🍘|🍚|🍙|🍛|🍬|🍰|🍣|🍡|🎀|🍷|🍻|🍾|👉|🎁|👑|🎈|🙄|🎂|🍩|😷|😎|😤|🎃|🎅|🎆|🎇|😃|😂|🎋|💃|🎉|🎊|🎒|🎗|🎩|🎬|🎥|🎭|🎯|🔓|🎲|🎷|🎺|🎻|🎸|🎾|🏀|🚲|💄|👠|👞|👟|👗|👔|👕|👛|💼|🗼|🚆|🚣|🎢|🐽|🐍|🐢|🐦|🐧|🐛|🐔|🐘|🐥|🐝|😰|😒|🐑|↖️|↖|🔣|⬆️|⬆|🏸👋|👋|💖|💙|💦|😻|🏾|👨|😩|🏿|🐀|🐋|👃|🐕|🐟|🐠|🐣|🐬|🐭🐳|🐵|👆|👈|🐼|👅|👺|👹|👻|💣|🐾|👀|😑|💛|👄|👎|😙|😗|🐻|💍|👊|💪|😀|👐|🖖|🍔|🍸|🍟|🌭|🍗|🌮|🛀|🌶|🏚|🤒|😆|👦|👙|👒|👖|👚|🐂|🍖|🍧|🍨|🍦|🍮|🍤|🐤|🕖|🔐|🔏|🔒|👣|👤|🌪|🖕|🔛|🔚|🔙|🔜|👥|👶|👁|💑|👧|👘|💏|👩|👪|👫|👯|👰|👳|👴|👵|👷|👬|👭|👲|👱|👼|🎵|🐐|🐓|🐰|🐇|🍹|🍺|🍝|🍲|🍜|🍈|👽|👾|👿|😇|💋|🔫|🔪|💐|✈|🚁|💔|💗|💕|💘|🏮|󞐊|💜|💟|💠|💢|😱|💩|🚒|🚧|🚚|⛽️|⛽|🚤|⛵️|⛵|🚊|🚉|🚠|⚓️|⚓|🚟|🚢|🏥|🍞|🎹|⚾️|⚾|🏇|🚵|🚅|🚄|🚂|🚝|🚈|📷|💬|💭|💮|🔟|💰|💵|🏩|📞|💳|💷|💸|💿|📃|📌|📙|📚|📒|📘|📗|🚪|💱|💶|💴|💻|🎮|🎰|📜|📝|📢|🖥|📿|🔄|🔆|🗝|🔝|🔥|🔰|🔱|🔸|🔹|🕋|🕗|🕥|🕚|⬆|⬅|⏫|↖|🖊|🖋|🚶|😀|😺|🤗|😕|😬|🤘|🙆|😅|😈|😊|😋|😍|🤓|🌬|😯|🙃|😢|😓|😔|😠|😖|😟|🙂|😛|😦|⚰|😣|😶|😶|🐈|😥|😧|😨|😪|😫|😮|😹|😸|😽|🙀|🙁|🙈|🙉|🙋|🙏|🚀|🚬|🏄|🏃|🚌|🚞|🚙|🏂|⚾|🗻|🌆|🗽|⛳|🎣|⛲|🏯|⛪|🏫|🚜|🏦|🏢|🚡|🏣|🛅|🏧|🌌|🌃|🛠|⛏|🏺|🔮|🤑|🤔|🤕|󞐼|󞐼|󾭉|󾭠 ", "",line)

    return line

# ff = clean_text(line="dfdjkfljlkسلاممممممممممممم")
#
# print(ff)
