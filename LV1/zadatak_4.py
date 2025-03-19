#Zadatak #4

#Napišite Python skriptu koja će učitati tekstualnu datoteku naziva song.txt.
#Potrebno je napraviti rječnik koji kao ključeve koristi sve različite riječi koje se pojavljuju u
#datoteci, dok su vrijednosti jednake broju puta koliko se svaka riječ (ključ) pojavljuje u datoteci.
#Koliko je riječi koje se pojavljuju samo jednom u datoteci? Ispišite ih.

def count_words_in_file(filename):
    word_count = {}

    with open(filename, 'r') as file:
        for line in file:
            words = line.lower().split()
            for word in words:
                word = word.strip(",.!?")
                if word:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
    return word_count

def find_single_occurrence_words(word_count):
    single_occurrence_words = []
    for word, count in word_count.items():
        if count == 1:
            single_occurrence_words.append(word)
    return single_occurrence_words

def main():
    filename = "song.txt"
    word_count = count_words_in_file(filename)
    single_occurrence_words = find_single_occurrence_words(word_count)

    print("Words that appear only once:", single_occurrence_words)
    print("Total number of words that appear only once:", len(single_occurrence_words))

if __name__ == "__main__":
    main()
