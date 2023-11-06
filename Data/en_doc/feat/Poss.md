---
layout: feature
title: 'Poss'
shortdef: 'possessive'
udver: '2'
---

In English, `Poss` is a Boolean feature of [pronouns](en-pos/PRON). It tells whether the word is possessive.

### <a name="Yes">`Yes`</a>: it is possesive

#### Examples

The following English pronouns have this feature:

* _my, mine, your, yours, his, her_ (if it has the PTB tag `PRP$`)_, hers, its, our, ours, their, theirs, whose_

Of these, the *dependent* ones (_my_, _your_, etc., which typically attach as [nmod:poss]()) also receive [Case]()=`Gen`.

Note that there is no `No` value. If the word is not possessive, the `Poss` feature will just not be mentioned in the `FEAT` column.
<!-- Interlanguage links updated Út 9. května 2023, 20:03:46 CEST -->
