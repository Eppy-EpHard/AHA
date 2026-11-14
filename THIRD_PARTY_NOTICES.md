# Third-party notices

**weight-space-lp** by Nathan Barry (https://github.com/nathanrs/weight-space-lp, commit a7de510), MIT licence
below. Parts of `src/afwsp/` are adapted from it, keeping only what this pipeline uses: the loop detector
(`loops.py`), the model hooks and checkpoint builder (`model.py`), the activation collection and the PDHG
solver for the LP (`edit.py`), and the maths evaluation harness (`evals.py`). `data/splits.json` is his split
of the practice problems.

The grading in `evals.py` is in turn ported from `exp8_math/code/mathqa.py` of *Sparks of Weight-Space Programming*
(Takehi, Teoh, Zhang and Khattab; MIT). The practice problems `scripts/setup.sh` downloads come from https://github.com/RikiyaT/lm_understand-logs.

```
MIT License

Copyright (c) 2026 Nathan Barry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
