# Third-party notices

The named-keypoint matching, SQPnP / RANSAC / positive-depth rejection / LM pipeline
in `pose.py` is adapted from
[HKUST ENTERPRIZE RM2026-Engineer-Assembly-Algorithm](https://github.com/hkustenterprize/RM2026-Engineer-Assembly-Algorithm/blob/7fb7609aa6b5a8b1992d32e14bc17d24d1a15578/runtime/core/arm_exchange_core/perception/pose.py),
revision `7fb7609aa6b5a8b1992d32e14bc17d24d1a15578`.
The upstream license follows. Other modules in this package are new implementations.

Version 0.3 adds local changes: supplemental AP3P hypotheses, explicit small-set
leave-one-out candidates, full-observation truncated scoring, repeated consensus
refinement and a Huber-weighted final fit. These improvements are not claimed to
be part of the upstream implementation. The benchmark loads the original solver
without changing its algorithm; only its observation type dependency is stubbed.

MIT License

Copyright (c) 2026 香港科技大学ENTERPRIZE战队（HKUST ENTERPRIZE Team）

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
