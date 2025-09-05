# -*- coding: utf-8 -*-
patterns = [
    (r'TRO[İI]D', 'TİROİD'),
    (r'T[İI]ROD', 'TİROİD'),
    (r'\bH[İI]PER[ A-ZÇĞİÖŞÜ]*T[İI]RO[İI]D[ A-ZÇĞİÖŞÜ]*\b', 'HİPERTİROİDİZM'),
    (r'\bH[İI]PER[T ]?R[İI]O[İI]D[ A-ZÇĞİÖŞÜ]*\b', 'HİPERTİROİDİZM'),
    (r'\bH[İI]PO[ A-ZÇĞİÖŞÜ]*T[İI]RO[İI]D[ A-ZÇĞİÖŞÜ]*\b', 'HİPOTİROİDİZM'),
    (r'\bH[İI]POR[ A-ZÇĞİÖŞÜ]*T[İI]RO[İI]D[ A-ZÇĞİÖŞÜ]*\b', 'HİPOTİROİDİZM'),
    (r'\bH[İI]PER[ A-ZÇĞİÖŞÜ]*T[İI]RO[İI]D[İI]\b', 'HİPERTİROİDİZM'),
    (r'\bH[İI]PO[ A-ZÇĞİÖŞÜ]*T[İI]RO[İI]D[İI]\b', 'HİPOTİROİDİZM'),
    (r'H[İI]PERT[İI]ROD[İI]ZM', 'HİPERTİROİDİZM'),
    (r'H[İI]POT[İI]ROD[İI]ZM', 'HİPOTİROİDİZM'),
    (r'H[İI]PORT[İI]RO[İI]D[İI]ZM', 'HİPOTİROİDİZM'),
    (r'H[İI]POT[İI]RO[İI]D[İI]ZM', 'HİPOTİROİDİZM'),
]

