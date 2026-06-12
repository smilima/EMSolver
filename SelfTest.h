//---------------------------------------------------------------------------
// SelfTest.h - headless solver verification (run with: RFSimulator -selftest)
//---------------------------------------------------------------------------
#ifndef SelfTestH
#define SelfTestH

// Runs physics sanity checks on the TLM solver and writes selftest.txt
// next to the executable. Returns 0 on success, 1 on failure.
int RunSelfTest();

#endif
