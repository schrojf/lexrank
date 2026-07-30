"""English language definition."""

from __future__ import annotations

from .base import Language
from .stemming import porter_stem

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to
too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
also just now said says get got may might must many much upon
""".split())

ABBREVIATIONS = frozenset("""
mr mrs ms mx dr prof sr jr st rev hon gen col maj capt lt sgt gov pres supt
inc ltd co corp llc plc dept div est fig figs no nos vol vols pp ed eds
approx appt apt assn assoc ave blvd rd sq dist univ inst
jan feb mar apr jun jul aug sept sep oct nov dec
mon tue tues wed thu thur thurs fri sat sun
etc vs viz cf al ca circa eg ie
a.m p.m u.s u.k u.s.a e.g i.e ph.d m.d b.a m.a b.s m.s d.c
""".split())

ENGLISH = Language(
    code="en",
    name="English",
    stopwords=STOPWORDS,
    abbreviations=ABBREVIATIONS,
    fold_diacritics=False,
    stemmer=porter_stem,
)
