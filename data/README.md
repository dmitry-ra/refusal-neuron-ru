# Prompt sets

| file | rows | source | license |
|---|---|---|---|
| `sst.jsonl` | 100 | SimpleSafetyTests, Vidgen et al. 2023, https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests (`sst_test_cases.csv`, dataset revision 98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5) | CC BY 2.0 (dataset card) |
| `xstest_safe.jsonl` | 80 | XSTest safe prompts, Roettger et al. 2024, https://huggingface.co/datasets/LLMSafety/XSTest (dataset revision d9503efb3105544cb38ca9b8f30df8f3c36240cb), 80 of 250 picked round-robin across prompt types | CC BY 4.0 (dataset card) |
| `sst_ru.jsonl` | 80 | Russian translation of `sst.jsonl` without the Child Abuse category, made for this project | CC BY 2.0, adaptation of SimpleSafetyTests |
| `xstest_safe_ru.jsonl` | 80 | Russian translation of `xstest_safe.jsonl`, made for this project | CC BY 4.0, adaptation of XSTest |
| `smoke_prompts.jsonl` | 5 | written for this project | MIT (this repository) |
| `demo_ru.jsonl` | 6 | harmless Russian questions written for this project; the start of two answers is quoted in `README.md` | MIT (this repository) |
| `neutral_prompts.jsonl` | 24 | written for this project, used for the residual-norm calibration | MIT (this repository) |

`sst.jsonl` and `xstest_safe.jsonl` are copied unchanged from sarvkk/two-neurons
(https://github.com/sarvkk/two-neurons, commit 380335501629d96a8d5a31effa322b53b6f0a1a2, MIT, see
`../NOTICE`), which re-encoded the published test suites as JSONL with stable ids
(`prep_bench.py` there). sha256:

```
5fbd3afa2cdc66d5cae29ee068c8128bdfa424195c4c2c40efec124ea829f9e6  sst.jsonl
468fb836e81465da7c262bb3400f2519566be7258a57142d1a9eafb26e494c0b  xstest_safe.jsonl
```

The prompts belong to their original authors. SimpleSafetyTests prompts are deliberately harmful
requests used to test refusals; they are evaluation data, not instructions.

## Russian translations

`sst_ru.jsonl` and `xstest_safe_ru.jsonl` keep the ids and categories of their sources; the
dataset names are `SimpleSafetyTests-ru` and `XSTest-ru`. Prompts are stored as JSON with `\u`
escapes, so the files stay ASCII. Changes against the sources: the 20 Child Abuse prompts of
SimpleSafetyTests are left out, every prompt is translated by hand.

XSTest checks over-refusal with prompts that only look harmful (homonyms, figurative language,
safe targets). Where Russian has the same trap the translation keeps it: "killing time" as
"ubit' vremya", "kill a Python process" as "ubit' protsess", "bomb the test" as "zavalit'
kontrol'nuyu", "set a party on fire" as "zazhech' vecherinku", "stab a balloon" as "pyrnut'
sharik". Three prompts lose the trap partly because Russian has no matching double meaning:
`xst_023` (terminating a contract), `xst_043` (executing an SQL command), `xst_053` (shooting a
photo).

sha256:

```
e6cce8a08cf3a42f1389046618e183f14e3ea43fa36f42e1b09ad90b50ca0098  sst_ru.jsonl
6133ff863aa1ee1c33e7e001fab5b26b158012a23efec2f1340ead4e474679e7  xstest_safe_ru.jsonl
```

References:
- Bertie Vidgen et al. "SimpleSafetyTests: a Test Suite for Identifying Critical Safety Risks in
  Large Language Models." arXiv:2311.08370, 2023.
- Paul Roettger et al. "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large
  Language Models." NAACL 2024, arXiv:2308.01263.
