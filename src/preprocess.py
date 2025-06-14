# src/preprocess.py

import os
import json

# 1) Mappning mellan boolean-nycklar (i dataset.jsonl) och de etiketter/labels vi vill använda
BOOLEAN_KEYS = [
    { "key": "specialistvård",         "label": "specialistvård" },
    { "key": "sjukhusvård",            "label": "sjukhusvård" },
    { "key": "operationer",            "label": "operationer" },
    { "key": "psykologbesök",          "label": "psykologbesök" },
    { "key": "fysioterapi",            "label": "fysioterapi" },
    { "key": "tandvård",               "label": "tandvård" },
    { "key": "receptbelagd_medicin",   "label": "receptbelagda mediciner" },
    { "key": "graviditetsrelaterad_vård", "label": "vård relaterad till graviditet" },
    { "key": "rehabilitering",         "label": "rehabilitering" },
    { "key": "vård_kroniska_sjukdomar","label": "vård av kroniska sjukdomar" },
    { "key": "cancerbehandlingar",     "label": "cancerbehandlingar" },
    { "key": "neurologiska_sjukdomar", "label": "vård vid neurologiska sjukdomar" },
    { "key": "logoped_talterapi",      "label": "logoped-/talterapi" },
    { "key": "smärtbehandling",        "label": "smärtbehandling" },
    { "key": "snabb_vårdgaranti_inom_7d",   "label": "snabb vårdgaranti inom 7 dagar" },
    { "key": "väntetid_specialist_inom_14d","label": "väntetid till specialist inom 14 dagar" },
    { "key": "väntetid_operation_inom_30d", "label": "väntetid till operation inom 30 dagar" },
    { "key": "läkarkontakt_inom_24h",       "label": "kontakt med läkare inom 24 timmar" },
    { "key": "vårdgaranti_vid_försening",   "label": "vårdgaranti vid försening" },
    { "key": "ersättning_vid_vårdgarantibrott","label": "ersättning vid vårdgarantibrott" },
    { "key": "eu_ees",                  "label": "vård inom EU/EES" },
    { "key": "norden",                  "label": "vård inom Norden" },
    { "key": "usa",                     "label": "vård i USA" },
    { "key": "asien",                   "label": "vård i Asien" },
    { "key": "australien",              "label": "vård i Australien" },
    { "key": "utan_hälsodeklaration",   "label": "teckning utan hälsodeklaration" },
    { "key": "bindningstid_1år",        "label": "en bindningstid på 1 år" },
    { "key": "teckna_efter_65år",       "label": "tecknas efter 65 år" },
    { "key": "ersättning_förlorad_arbetsinkomst","label": "ersättning för förlorad arbetsinkomst" },
    { "key": "second_opinion",          "label": "möjlighet till second opinion" },
    { "key": "privatläkare",            "label": "privatläkarkontakt" },
    { "key": "digital_vård",            "label": "digital vård" },
    { "key": "separat_barnförsäkring",  "label": "en separat barnförsäkring" },
    { "key": "täcker_vaccinationer",    "label": "vaccinationer" },
    { "key": "täcker_hjälpmedel_funktionsnedsättning", "label": "hjälpmedel vid funktionsnedsättning" },
    { "key": "karenstid",               "label": "en karenstid" },
]

# 2) FAQ-frågemallar per label
FAQ_TEMPLATES = {
  "specialistvård": [
    "Ersätter försäkringen kostnaderna för specialistvård?",
    "Omfattar försäkringen specialistvård?",
    "Erbjuder försäkringen direkt tillgång till specialistvård?"
  ],
  "sjukhusvård": [
    "Ersätter försäkringen kostnaderna för sjukhusvård?",
    "Omfattar försäkringen sjukhusvård?",
    "Erbjuder försäkringen sjukhusvård?"
  ],
  "operationer": [
    "Ersätter försäkringen kostnaderna för operationer?",
    "Omfattar försäkringen operationer?",
    "Erbjuder försäkringen operationer?"
  ],
  "psykologbesök": [
    "Ersätter försäkringen kostnaderna för psykologbesök?",
    "Omfattar försäkringen psykologbesök?",
    "Erbjuder försäkringen psykologbesök?"
  ],
  "fysioterapi": [
    "Ersätter försäkringen kostnaderna för fysioterapi?",
    "Omfattar försäkringen fysioterapi?",
    "Erbjuder försäkringen fysioterapi?"
  ],
  "tandvård": [
    "Ersätter försäkringen kostnaderna för tandvård?",
    "Omfattar försäkringen tandvård?",
    "Erbjuder försäkringen tandvård?"
  ],
  "receptbelagda mediciner": [
    "Ersätter försäkringen kostnaderna för receptbelagda mediciner?",
    "Omfattar försäkringen receptbelagda mediciner?",
    "Erbjuder försäkringen ersättning för receptbelagda mediciner?"
  ],
  "vård relaterad till graviditet": [
    "Ersätter försäkringen kostnaderna för vård relaterad till graviditet?",
    "Omfattar försäkringen graviditetsrelaterad vård?",
    "Erbjuder försäkringen vård relaterad till graviditet?"
  ],
  "rehabilitering": [
    "Ersätter försäkringen kostnaderna för rehabilitering?",
    "Omfattar försäkringen rehabilitering?",
    "Erbjuder försäkringen rehabilitering?"
  ],
  "vård av kroniska sjukdomar": [
    "Ersätter försäkringen kostnaderna för vård vid kroniska sjukdomar?",
    "Omfattar försäkringen vård för kroniska sjukdomar?",
    "Erbjuder försäkringen vård för kroniska sjukdomar?"
  ],
  "cancerbehandlingar": [
    "Ersätter försäkringen kostnaderna för cancerbehandlingar?",
    "Omfattar försäkringen cancerbehandlingar?",
    "Erbjuder försäkringen cancerbehandlingar?"
  ],
  "vård vid neurologiska sjukdomar": [
    "Ersätter försäkringen kostnaderna för behandling av neurologiska sjukdomar?",
    "Omfattar försäkringen vård vid neurologiska sjukdomar?",
    "Erbjuder försäkringen behandling vid neurologiska sjukdomar?"
  ],
  "logoped-/talterapi": [
    "Ersätter försäkringen kostnaderna för logoped- eller talterapi?",
    "Omfattar försäkringen logoped- eller talterapi?",
    "Erbjuder försäkringen logoped- eller talterapi?"
  ],
  "smärtbehandling": [
    "Ersätter försäkringen kostnaderna för smärtbehandling?",
    "Omfattar försäkringen smärtbehandling?",
    "Erbjuder försäkringen smärtbehandling?"
  ],
  "snabb vårdgaranti inom 7 dagar": [
    "Erbjuder försäkringen en snabb vårdgaranti inom 7 dagar?",
    "Omfattar försäkringen snabb vårdgaranti inom 7 dagar?",
    "Ersätter försäkringen kostnaderna om snabb vårdgaranti inte infrias inom 7 dagar?"
  ],
  "väntetid till specialist inom 14 dagar": [
    "Erbjuder försäkringen specialistvård med en väntetid på under 14 dagar?",
    "Omfattar försäkringen snabb specialistvård med en väntetid på mindre än 14 dagar?",
    "Ersätter försäkringen kostnaderna om väntetiden för specialistvård överstiger 14 dagar?"
  ],
  "väntetid till operation inom 30 dagar": [
    "Erbjuder försäkringen operationer med en väntetid på under 30 dagar?",
    "Omfattar försäkringen operationer med kort väntetid (under 30 dagar)?",
    "Ersätter försäkringen kostnaderna om väntetiden för operationer överstiger 30 dagar?"
  ],
  "kontakt med läkare inom 24 timmar": [
    "Erbjuder försäkringen kontakt med en läkare inom 24 timmar?",
    "Omfattar försäkringen snabb läkarkontakt inom 24 timmar?",
    "Ersätter försäkringen kostnaderna om du inte får kontakt med en läkare inom 24 timmar?"
  ],
  "ersättning vid vårdgarantibrott": [
    "Ersätter försäkringen kostnader vid brott mot vårdgarantin?",
    "Omfattar försäkringen ersättning vid vårdgarantibrott?",
    "Erbjuder försäkringen kompensation vid vårdgarantibrott?"
  ],
  "vård inom EU/EES": [
    "Omfattar försäkringen vård i EU/EES?",
    "Erbjuder försäkringen ersättning för vårdkostnader i EU/EES?",
    "Ersätter försäkringen kostnaderna för vård utomlands inom EU/EES?"
  ],
  "vård inom Norden": [
    "Omfattar försäkringen vård i Norden?",
    "Erbjuder försäkringen ersättning för vård i Norden?",
    "Ersätter försäkringen kostnaderna för vård i Norden?"
  ],
  "vård i USA": [
    "Omfattar försäkringen vård i USA?",
    "Erbjuder försäkringen ersättning för vård i USA?",
    "Ersätter försäkringen kostnaderna för vård i USA?"
  ],
  "vård i Asien": [
    "Omfattar försäkringen vård i Asien?",
    "Erbjuder försäkringen ersättning för vård i Asien?",
    "Ersätter försäkringen kostnaderna för vård i Asien?"
  ],
  "vård i Australien": [
    "Omfattar försäkringen vård i Australien?",
    "Erbjuder försäkringen ersättning för vård i Australien?",
    "Ersätter försäkringen kostnaderna för vård i Australien?"
  ],
  "teckning utan hälsodeklaration": [
    "Kan försäkringen tecknas utan hälsodeklaration?",
    "Omfattar försäkringen möjligheten att tecknas utan hälsodeklaration?",
    "Erbjuder försäkringen alternativ utan krav på hälsodeklaration?"
  ],
  "en bindningstid på 1 år": [
    "Har försäkringen en bindningstid på 1 år?",
    "Omfattar försäkringen en bindningstid på 1 år?",
    "Erbjuder försäkringen en 1-årig bindningstid?"
  ],
  "tecknas efter 65 år": [
    "Kan försäkringen tecknas efter 65 år?",
    "Omfattar försäkringen möjligheten att tecknas efter 65 år?",
    "Erbjuder försäkringen teckning även efter 65 år?"
  ],
  "ersättning för förlorad arbetsinkomst": [
    "Ersätter försäkringen förlorad arbetsinkomst?",
    "Omfattar försäkringen ersättning för förlorad arbetsinkomst?",
    "Erbjuder försäkringen kompensation för förlorad arbetsinkomst?"
  ],
  "möjlighet till second opinion": [
    "Erbjuder försäkringen möjlighet till second opinion?",
    "Omfattar försäkringen second opinion?",
    "Ersätter försäkringen kostnaderna för second opinion?"
  ],
  "privatläkarkontakt": [
    "Omfattar försäkringen privatläkare?",
    "Erbjuder försäkringen tillgång till privatläkare?",
    "Ersätter försäkringen kostnaderna för privatläkare?"
  ],
  "digital vård": [
    "Erbjuder försäkringen digital vård?",
    "Omfattar försäkringen digital vård?",
    "Ersätter försäkringen kostnaderna för digital vård?"
  ],
  "en separat barnförsäkring": [
    "Finns en separat barnförsäkring inkluderad?",
    "Omfattar försäkringen även en separat barnförsäkring?",
    "Erbjuder försäkringen en separat barnförsäkring?"
  ],
  "vaccinationer": [
    "Ersätter försäkringen kostnaderna för vaccinationer?",
    "Omfattar försäkringen vaccinationer?",
    "Erbjuder försäkringen vaccinationer?"
  ],
  "hjälpmedel vid funktionsnedsättning": [
    "Ersätter försäkringen kostnaderna för hjälpmedel vid funktionsnedsättning?",
    "Omfattar försäkringen hjälpmedel vid funktionsnedsättning?",
    "Erbjuder försäkringen hjälpmedel vid funktionsnedsättning?"
  ],
  "en karenstid": [
    "Finns det en karenstid?",
    "Omfattar försäkringen en karenstid?",
    "Erbjuder försäkringen teckning utan karenstid?"
  ],
}

# 3) Svarsmallar beroende på frågeord (”ersätter”, ”omfattar”, ”erbjuder”, ”har”, ”kan”):
ANSWER_TEMPLATES = {
  "ersätter": {
    "yes": "Ja, {nivå} ersätter kostnaderna för {label}.",
    "no":  "Nej, {nivå} ersätter inte kostnaderna för {label}."
  },
  "omfattar": {
    "yes": "Ja, {nivå} omfattar {label}.",
    "no":  "Nej, {nivå} omfattar inte {label}."
  },
  "erbjuder": {
    "yes": "Ja, {nivå} erbjuder {label}.",
    "no":  "Nej, {nivå} erbjuder inte {label}."
  },
  "har": {
    "yes": "Ja, {nivå} har {label}.",
    "no":  "Nej, {nivå} har inte {label}."
  },
  "kan": {
    "yes": "Ja, {nivå} kan {label}.",
    "no":  "Nej, {nivå} kan inte {label}."
  }
}

def detect_pattern(question: str) -> str:
    """
    Avgör vilket ”mönster” (ersätter/omfattar/erbjuder/har/kan) som frågan använder.
    Utifrån detta kan vi välja rätt svarsmall.
    """
    q = question.lower()
    if q.startswith("har "):
        return "har"
    if q.startswith("kan "):
        return "kan"
    if q.startswith("finns"):
        return "har"
    if "ersätter" in q:
        return "ersätter"
    if q.startswith("omfattar"):
        return "omfattar"
    if q.startswith("erbjuder"):
        return "erbjuder"
    # Fallback
    return "erbjuder"

def preprocess():
    """
    Läser in data/dataset.jsonl och skriver ut data/dataset_faq.jsonl
    där varje rad är en FAQ-fråga (input) och ett svar (output).
    """
    src_path = os.path.join("data", "dataset.jsonl")
    dst_path = os.path.join("data", "dataset_faq.jsonl")

    # Läs in alla poster från dataset.jsonl
    with open(src_path, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    # Skriv ut varje FAQ‐entry
    with open(dst_path, "w", encoding="utf-8") as out:
        for entry in entries:
            nivå = entry["försäkring"]  # t.ex. "Folksam Bas"
            for bk in BOOLEAN_KEYS:
                key   = bk["key"]
                label = bk["label"]
                value = entry.get(key, False)

                # Hitta motsvarande FAQ-mall i FAQ_TEMPLATES
                if label not in FAQ_TEMPLATES:
                    continue

                for question in FAQ_TEMPLATES[label]:
                    pat = detect_pattern(question)
                    ans = ANSWER_TEMPLATES[pat]["yes" if value else "no"]
                    answer = ans.format(nivå=nivå, label=label)

                    faq_entry = {
                        "input":  question,
                        "output": answer
                    }
                    out.write(json.dumps(faq_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    preprocess()
