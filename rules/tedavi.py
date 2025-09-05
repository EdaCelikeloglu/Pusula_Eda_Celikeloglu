# -*- coding: utf-8 -*-
# rules/tedavi.py
# TedaviAdi sütununu kanonikleştirmek için regex -> replacement çiftleri

patterns = [
    # Basit normalize: " -1/-2" gibi seviye/versiyon son eklerini at
    (r'[-–—]\s*\d+\s*$', ''),

    # TR/diakritik varyantları zaten common.py ile toparlanıyor;
    # burada alan-özel toparlamaları yapıyoruz.

    # DORSALJİ: sonradan gelen bölge eklerini at, tek başlıkta topla
    (r'\bDORSALJ[İI].*', 'DORSALJİ'),

    # OMUZ İMPİNGEMENT: sağ/sol, post-op, eklentiler → tek başlık
    (r'\b(SAĞ|SOL)\s+OMUZ\s+İMP[İI]N?G?E?MENT\b', 'OMUZ İMPİNGEMENT'),
    (r'\bOMUZ\s+İMP[İI]N?G?E?MENT\b',             'OMUZ İMPİNGEMENT'),
    (r'\bİMP[İI]N?G?E?MENT\s+(SAĞ|SOL)\b',         'OMUZ İMPİNGEMENT'),
    (r'\bİMP[İI]NGEMENT\b',                        'OMUZ İMPİNGEMENT'),

    # İNTERVERTEBRAL DİSK BOZUKLUĞU varyantları
    (r'\b[İI]V\s*D[İI]SK\s+BOZUKLU[ĞG]U\s*-\s*BEL\b.*',       'İNTERVERTEBRAL DİSK BOZUKLUĞU - BEL'),
    (r'\b[İI]V\s*D[İI]SK\s+BOYUN\b',                          'İNTERVERTEBRAL DİSK BOZUKLUĞU - SERVİKAL'),
    (r'\b[İI]V\s*D[İI]SK\s+BOZUKLU[ĞG]U\b',                   'İNTERVERTEBRAL DİSK BOZUKLUĞU'),
    (r'\bSERV[İI]KAL\s+D[İI]SK\s+HERN[İI]S[İI]\b',            'SERVİKAL DİSK HERNİSİ'),

    # Gonartroz / Meniskopati birlikte yazımlar
    (r'\bGONARTROZ[-/+\s]*MEN[İI]SKOPAT[İI]\b',               'GONARTROZ/MENİSKOPATİ'),

    # Aşil / Patella / Malleol / Metakarp vb. temel başlıklar
    (r'\bA[ŞS][İI]L\s+K[ıİ]SAL[ıİ][ğĞ][ıİ]\b',               'KISA AŞİL TENDONU'),
    (r'\bA[ŞS][İI]L\s+R[ÜU]PT[ÜU]R[ÜU].*',                   'AŞİL RÜPTÜRÜ'),
    (r'\bA[ŞS][İI]L\s+TEND[İI]N[İI]T[İI].*',                  'AŞİL TENDİNİT'),
    (r'\bPATELLA\s+KIRI[ĞG][İI]\b',                           'PATELLA KIRIĞI'),
    (r'\bMALLEOL\s+KIRI[ĞG][İI]\b',                           'MALLEOL KIRIĞI'),
    (r'\bMETAKARP(?:AL)?\s+KIRI[ĞG][İI]\b',                   'METAKARP KIRIĞI'),

    # Rehabilitasyon söylemleri → tek tipe indir
    (r'\bREHAB[İI]L[İI]TASYON(U|U)?\b',                       'REHABİLİTASYON'),
    (r'\bREH\b',                                              'REHABİLİTASYON'),
    (r'\bOP\s*[- ]?\s*İZOMETR[İI]K\b',                        'OP İZOMETRİK'),
    (r'\bPOST\s*OP\b',                                        'POSTOP'),

    # EL / EL BİLEĞİ tedavileri
    (r'\bEL\s+REHAB[İI]L[İI]TASYONU\b',                       'EL REHABİLİTASYONU'),
    (r'\bEL\s+REHAB[İI]L[İI]TASYON\s*PROGRAMI\b',             'EL REHABİLİTASYONU'),
    (r'\bEKSTANS[ÖO]R\s+TENDON\s+REHAB[İI]L[İI]TASYON\b',     'EKSTANSÖR TENDON REHABİLİTASYONU'),
    (r'\bTFCC\s*REHAB[İI]L[İI]TASYON(U|U)?\b',                'TFCC REHABİLİTASYONU'),

    # Nöro / pediatrik şablonlar
    (r'\bSEREBRAL\s+PALS[İI]\b',                              'SEREBRAL PALSİ'),
    (r'\bPED[İI]ATR[İI]K\s+REHAB[İI]L[İI]TASYON\s*PROGRAMI\b','PEDİATRİK REHABİLİTASYON PROGRAMI'),
    (r'\bPARAPLEJ[İI][-–—/]\s*TETRAPLEJ[İI]\b',               'PARAPLEJİ/TETRAPLEJİ'),
    (r'\bHEM[İI]PAR(E|A)Z[İI]\b',                             'HEMİPAREZİ'),
    (r'\bHEM[İI]PLEJ[İI]\b',                                  'HEMİPLEJİ'),

    # Kas-iskelet genel
    (r'\bMUSKUL(?:ER|AR)\s+STRA[İI]N\b',                      'KAS ZORLANMASI'),
    (r'\bMYO(?:DASCI|FASC)AL\s+AĞRI\b',                       'MİYOFASİYAL AĞRI'),
    (r'\bPLANTAR\s+FAS[İI][İI]T\b',                           'PLANTAR FASİİT'),
    (r'\bTEND[İI]N[İI]T[- ]?TENOS[İI]NOV[İI]T\b',             'TENDİNİT/TENOSİNOVİT'),

    # Koksartroz / Skolyoz / Kifoz vb.
    (r'\bKOKSARTROZ\d*\b',                                    'KOKSARTROZ'),
    (r'\bSKOLYOZ\b',                                          'SKOLYOZ'),
    (r'\bK[İI]FOZ\b',                                         'KİFOZ'),

    # Artan birleşik yazımlar
    (r'\bKALKANEAL\s+SPUR\b.*',                               'KALKANEAL SPUR'),
    (r'\bTROKANTER[İI]K\s+BURS[İI]T\b',                       'TROKANTERİK BURSİT'),

    # Çeşitli yazım/çoğulluk düzeltmeleri
    (r'\bD[İI]Z\s+OP\s*-?\s*ERKEN\s+REHAB[İI]L[İI]TASYON\b',  'DİZ OP ERKEN REHABİLİTASYON'),
    (r'\bY[ÜU]R[ÜU]ME\s*E[ğĞ][İI]T[İI]M[İI]\b',              'YÜRÜME EĞİTİMİ'),
    (r'\bORGAN\s+NAKL[İI]\s+MOB[İI]L[İI]ZASYON\b',            'ORGAN NAKLİ MOBİLİZASYON'),
]
