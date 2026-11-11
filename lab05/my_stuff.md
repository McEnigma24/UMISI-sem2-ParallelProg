map: Przyjmuje od razu całą listę argumentów. Zwraca wyniki ściśle w początkowej kolejności
(blokuje pętlę wyjściową, czekając na opóźniony wiersz, nawet jeśli późniejsze wiersze są już policzone).

submit: Zleca jedno zadanie i zwraca obiekt Future (obietnicę wyniku). Pozwala (w połączeniu z as_completed) odbierać wyniki w kolejności ich rzeczywistego ukończenia (kto pierwszy, ten lepszy), całkowicie eliminując przestoje w pętli for.