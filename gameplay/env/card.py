def card(rank, suit):
    return suit * 13 + ((rank - 2) % 13)

def rank(card):
    return (card % 13) + 2

def suit(card):
    return card // 13