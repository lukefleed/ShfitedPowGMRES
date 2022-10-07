# MAIL DI BINI

Buongiorno,

il progetto del corso di Calcolo Scientifico deve necessariamente intersecare gli argomenti sviluppati a lezione.

Le propongo allora questo progetto che tocca i metodi delle potenze, il processo di Arnoldi e il GMRES con restart e riguarda il page rank.

Nel lavoro che trova allegato (_autori Shen, Su, Carpentieri e Weng_)  sono introdotti e confrontati alcuni metodi per effettuare il calcolo del vettore PageRank di una stessa rete ma con diversi valori del parametro di damping. Gli autori motivano questo interesse scrivendo che questo problema computazionale si incontra nel progetto di dispositivi anti-spam e citano per questo il lavoro _P.G. Constantine , D.F. Gleich , Random alpha PageRank, Internet Math. 6 (2009) 189–236._

Nel lavoro allegato vengono introdotti e confrontati alcuni metodi per effettuare questo calcolo. Precisamente:

`Algorithm 1 (pag. 6):` Metodo delle potenze adattato al calcolo del PageRank con diversi valori del parametro di damping

`Algorithm 4 (pag. 10):` Metodo del GMRES adattato al calcolo del PageRank con diversi valori del parametro di damping. Questo algoritmo utilizza il metodo di Arnoldi  e il GMRES con restart che gli autori riportano per completezza rispettivamente nell'Algorithm 2 a pag. 7 e nell'Algorithm 3 a pag. 8.

Gli autori combinano poi `Algorithm 4` e `Algorithm 1` nell'`Algorithm 5` a pag. 10 per aumentare l'efficienza.

Il progetto riguarderebbe l'implementazione di `Algorithm 1` e `Algorithm 4` e un loro confronto per valutare l'efficienza in termini di numero di iterazioni e di tempo di cpu impiegato. Problemi test validi sono  la matrice web-Stanford e web-BerkStan.

Se nella descrizione degli algoritmi ci fossero dei punti non chiari, bisogna andare a leggere la parte di testo che descrive gli algoritmi stessi. Ad esempio nella linea 14 dell'algoritmo 4 compare un vettore z che gli autori non definiscono dentro l'algoritmo. Questo vettore compare nella formula (40) e infatti viene definito nella riga subito dopo la (40).

Mi faccia sapere se incontra difficolta' di comprensione degli algoritmi (talvolta gli autori non sono molto precisi nel riportare i loro risultati). Ci risentiamo poi per mettere a punto i test da fare.

A presto,
Dario Bini
