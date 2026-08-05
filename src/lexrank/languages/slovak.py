"""Slovak language definition.

Stopwords, abbreviations and ordinal followers are stored with diacritics
folded, because :class:`~lexrank.languages.base.Language` folds Slovak tokens
before any lookup happens.
"""

from __future__ import annotations

from .base import Language
from .stemming import slovak_stem

# Folded forms: "keď" -> "ked", "že" -> "ze", "sú" -> "su".
STOPWORDS = frozenset(
    """
a aby aj ak ako ale alebo ani ano asi az bez bol bola boli bolo bude budem
budes budeme budete budu by bym bys byt cez co coho com comu ci cim cize
dalej dalsi dnes do ho hoci i iba ich im ine inak iny ista iste isty ja je
jeho jej jemu jej ked kedy kedze keby kto ktora ktore ktoreho ktorej ktorom
ktorou ktory ktorych ktorym ktorymi ku k lebo len ma mal mala mali malo mate
medzi menej mi mna mne mnou moj moja moje mozno moze mozem mu my na nad nam
nas nasa nase nech nej nemu nich nie niektore nim nimi no nou o od odo on
ona oni ono ony po pod podla pokial popri potom pre pred predo preco preto
pretoze pri prave prece s sa sam sama same sami si sme so som ste su svoj
svoja svoje svojich svojim ta tak take takze taky tam tato teda tej tejto ten
tento teraz tie tieto tiez to toho tohto tom tomto tomu tomuto toto tu tuto
ty tym tymto tymi u uz v vam vas vasa vase viac vo vsak vsetci vsetko vsetky
vy z za zo ze zas este sice nielen nic nikto niekto vsade vtedy vzdy len
bola byt mat mozu museli musi maju
""".split()
)

ABBREVIATIONS = frozenset(
    """
napr atd atp apod tzv tzn resp pripadne popr cca str s r o a.s s.r.o
mil mld tis hod min sek kap obr tab vid pozri sv ul nam okr kraj
prof doc ing mgr bc phdr judr mudr rndr paeddr mvdr csc drsc phd arch dipl
tel fax mob email www c cislo ods pism cl zb zak vyd
jan feb mar apr maj jun jul aug sep sept okt nov dec
p pp t.j tj t.z napr. resp. atd. sl slov angl nem lat gr
""".split()
)

# Slovak writes dates and ordinals as "5. mája 1945" / "20. storočia" -- a
# period after a number followed by one of these must not end the sentence.
ORDINAL_FOLLOWERS = frozenset(
    """
januara februara marca aprila maja juna jula augusta septembra oktobra
novembra decembra januar februar marec april maj jun jul august september
oktober november december
storocia storocie storoci rocnika rocnik polrok polroka kvartal kvartalu
miesto mieste miesta poschodie poschodi triede trieda kola kolo kole
""".split()
)

SLOVAK = Language(
    code="sk",
    name="Slovak",
    stopwords=STOPWORDS,
    abbreviations=ABBREVIATIONS,
    ordinal_followers=ORDINAL_FOLLOWERS,
    fold_diacritics=True,
    stemmer=slovak_stem,
)
