# -*- coding: utf-8 -*-
patterns = [
    # Bozulmuş yazımlar
    (r'\bSERV\s*[İI]KO\s*TORAS[İI]K\s+BÖLGE\b', 'SERVİKOTORASİK BÖLGE'),
    (r'\bİNTERVERTEBRA\s*L\b', 'İNTERVERTEBRAL'),
    (r'\b[İI]\s*NTERVERTEBRAL\b', 'İNTERVERTEBRAL'),
    (r'\bSERV[İI]KAL\s*D[İI]SK\s*BOZUKLUKLAR\s*I\b', 'SERVİKAL DİSK BOZUKLUKLARI'),
    (r'\bMEN[İI]SK[ÜU]S\s*YIRTI\s*ĞI\b', 'MENİSKÜS YIRTIĞI'),
    (r'\bLATERAL\s*MALL\s*EOL\s*KIRI[ĞG][İI]\b', 'LATERAL MALLEOL KIRIĞI'),
    (r'\bPELV\s*IK\b', 'PELVİK'),
    (r'\bL[İI]GAMEN\s*TLER[İI]N\b', 'LİGAMENTLERİN'),
    (r'\bL[İI]GAM\s*E\s*NTLER[İI]N\b', 'LİGAMENTLERİN'),
    (r'\bPARMAK\s*LAR\s*IN\b', 'PARMAKLARIN'),
    (r'\bİNT[İI]R[İI]NS[İI]K\b', 'İNTRİNSİK'),
    (r'\b[İI]NTR[EA]HEPAT[İI]K\b', 'İNTRAHEPATİK'),

    # TR/EN karışık
    (r'\bKAS\s+ZORLANMASI\s+MUSCULAR\s+STRA[İI]N\b', 'KAS ZORLANMASI'),
    (r'\bROTATOR\s+KUF\b', 'ROTATOR CUFF'),

    # Roll-up
    (r'\bGONARTROZ\s+D[İI]Z\s+EKLEM[İI]N[İI]N\s+ARTROZU\b', 'GONARTROZ'),
    (r'\bKOKSARTROZ\s+KAL[ÇC]A\s+ARTROZU\b', 'KOKSARTROZ'),
    (r'\bPR[İI]MER\s+GONARTROZ\b', 'GONARTROZ'),
    (r'\bSEREBROVASK[ÜU]LER\s+HASTALIKLAR(?:\s+D[İI]ĞER)?\b', 'SEREBROVASKÜLER HASTALIK'),
    (r'\bSEREBROVASK[ÜU]LER\s+HASTALIK\s+D[İI]ĞER\b', 'SEREBROVASKÜLER HASTALIK'),
    (r'\bRAD[İI]K[ÜU]LOPAT[İI]\s+İLE\b', 'RADİKÜLOPATİ'),
    (r'\bPER[İI]TONEAL\s+APSE\s+İLE\b', 'PERİTONEAL APSE'),
    (r'\bD[İI]YABETES\s+MELL[İI]T[ÜU]S\b', 'DİYABET'),
    (r'\bV[İI]TAM[İI]NLER[İI]N\s+EKS[İI]KL[İI][ĞG][İI]\b', 'VİTAMİN EKSİKLİĞİ'),

    # Tendinit tekilleştirme
    (r'\bTEND[İI]N[İI]T[İI]\b', 'TENDİNİT'),
    (r'\bKALS[İI]F[İI]K\s+TEND[İI]N[İI]T[İI]\b', 'KALSİFİK TENDİNİT'),
    (r'\bOMUZUN\s+KALS[İI]F[İI]K\s+TEND[İI]N[İI]T\b', 'KALSİFİK TENDİNİT'),

    # ICD birlikte yazımlar
    (r'\b([A-ZÇĞİÖŞÜ]+(?:\s+[A-ZÇĞİÖŞÜ]+)*)\s+(?:[İI]LE\s+)?([A-Z]\d{2})\s+(\d)\b', r'\1 \2 \3'),

    # Konsept kısaltmalar ve sadeleştirmeler (örneklerden)
    (r'\bD[İI]Z\s+ANTER[İI]OR\s+POSTER[İI]OR\s+ÇAPRAZ\s+L[İI]GAMENT\s+BURKULMA\s+VE\s+GER[İI]LMES[İI]\b', 'DİZ ÇAPRAZ BAĞ BURKULMA/GERİLME'),
    (r'\bPATELLANIN\s+D[İI]ĞER\s+YERLEŞ[İI]M\s+BOZUKLUKLARI\b', 'PATELLA YERLEŞİM BOZUKLUKLARI'),
    (r'\bART[İI]K[ÜU]LER\s+KIKIRDAK\s+D[İI]ĞER\s+BOZUKLUKLARI\b', 'ARTİKÜLER KIKIRDAK BOZUKLUKLARI'),
    (r'\bS[İI]NOVYA\s+VE\s+TENDONUN\s+D[İI]ĞER\s+BOZUKLUKLARI\b', 'TENDON/SİNOVYA BOZUKLUKLARI'),

    # Lokasyon + işlem sadeleştirme
    (r'\bAYAK/?AYAK\s+B[İI]LE[ĞG][İI]\s+(?:D[ÜU]ZEY[İI]NDE\s+)?EKLEM\s+VE\s+L[İI]GAM(?:\s*E\s*N|\s*EN)?\s*TLER[İI]N[İI]?\s+ÇIKIK\b', 'AYAK/AYAK BİLEĞİ ÇIKIK'),
    (r'\bAYAK/?AYAK\s+B[İI]LE[ĞG][İI]\s+(?:D[ÜU]ZEY[İI]NDE\s+)?KAS\s+VE\s+TENDON\s+YARALANMASI\b', 'AYAK/AYAK BİLEĞİ KAS/TENDON YARALANMASI'),
    (r'\bEL\s+B[İI]LE[ĞG][İI]\s+VE\s+EL\s+D[ÜU]ZEY[İI]NDE\s+EKLEM\s+VE\s+L[İI]GAM(?:\s*E\s*N|\s*EN)?\s*TLER[İI]N[İI]?\s+ÇIKIK\b', 'EL/EL BİLEĞİ ÇIKIK'),

    # Senin eklediğin dönüşümler (örneklerinden)
    (r'\bEKLEMDE\s+AĞRI\b', 'EKLEM AĞRISI'),
    (r'\bALLERJ[İI]K\b', 'ALERJİK'),
    (r'\bKONDROMALAZ[İI]A\b', 'KONDROMALAZİ'),
    (r'\bRAD[İI]US\s+ÜST\s+UCU\b', 'RADİUS ÜST UÇ'),
    (r'\bKALSIFIK\b', 'KALSİFİK'),
    (r'\bTENDINIT\b', 'TENDİNİT'),

    # CRPS standardizasyonu
    (r'\bALGON[ÖO]ROD[İI]STROF[İI]\b', 'CRPS (KOMPLEKS BÖLGESEL AĞRI)'),


]
