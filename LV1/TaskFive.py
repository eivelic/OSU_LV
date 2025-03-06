#Zadatak #5

#Napišite Python skriptu koja će učitati tekstualnu datoteku naziva SMSSpamCollection.txt
#Ova datoteka sadrži 5574 SMS poruka pri čemu su neke označene kao spam, a neke kao ham.

#Primjer dijela datoteke:
#ham Yup next stop.
#ham Ok lar... Joking wif u oni...
#spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!

#a) Izračunajte koliki je prosječan broj riječi u SMS porukama koje su tipa ham, a koliko je
#prosječan broj riječi u porukama koje su tipa spam.
#b) Koliko SMS poruka koje su tipa spam završava uskličnikom ?

def main():
    ham_word_count = 0
    spam_word_count = 0
    ham_messages = 0
    spam_messages = 0
    spam_with_exclamation = 0

    with open('SMSSpamCollection.txt', 'r', encoding='utf-8') as file:
        for line in file:
            label, message = line.split('\t')
            words = message.split()

            if label == 'ham':
                ham_word_count += len(words)
                ham_messages += 1
            elif label == 'spam':
                spam_word_count += len(words)
                spam_messages += 1
                if message.endswith('!\n'):
                    spam_with_exclamation += 1

    avg_ham_words = ham_word_count / ham_messages if ham_messages else 0
    avg_spam_words = spam_word_count / spam_messages if spam_messages else 0

    print(f"Average words in ham messages: {avg_ham_words:.2f}")
    print(f"Average words in spam messages: {avg_spam_words:.2f}")
    print(f"Spam messages with exclamation mark: {spam_with_exclamation}")

if __name__ == "__main__":
    main()
