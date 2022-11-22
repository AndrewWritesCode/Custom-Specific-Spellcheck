# Custom-Spellcheck
Takes a list of words specified by user and spellchecks input text to that dictionary

Flow Chart of Valid Word Initialization
```mermaid
graph TD;
    A[valid word added]-->B[wordScore calculated];
    C[charScores used to determine character weights and dimensionality]-->B[wordScore calculated];
    B[wordScore calculated]-->D[wordScore and data associated with word stored in Valid Dictionary hash map];
    D[wordScore and data associated with word stored in Valid Dictionary hash map]-->E[WordBook];
    D-->A
    E[WordBook initialized]-->F[user input read];
    F[user input read]-->G[input wordScore calculated];
    G[input wordScore calculated]-->H[input wordScore compared to wordScores in Valid Dictionary];
    H[input wordScore compared to wordScores in Valid Dictionary]-->J[Valid Dictionary entry with closest wordScore match returned];
    J[Valid Dictionary entry with closest wordScore match returned]-->F[user input read];
```



<p align="center">
  <b>Example:</b><br>
  INPUT</b><br>
  rgb = ColorWordBook[spellcheck("ponk", ColorWordBook, charScores)]["RGB"]</b><br>
  OUTPUT</b><br>
  rgb = ColorWordBook["pink"]["RGB"]</b><br>
  <br><br>
</p>
