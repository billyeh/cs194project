typedef union {
    double d;
    unsigned short s[4];
} ieee754;

static const double LOG2E  =  1.4426950408889634073599;     /* 1/log(2) */
static const double C1 = 6.93145751953125E-1;
static const double C2 = 1.42860682030941723212E-6;

double exp_taylor5(double x)
{
    int n;
    double a, px;
    ieee754 u;

    /* n = round(x / log 2) */
    a = LOG2E * x + 0.5;
    n = (int)a;
    n -= (a < 0);

    /* x -= n * log2 */
    px = (double)n;
    x -= px * C1;
    x -= px * C2;

    /* Compute e^x using a polynomial approximation. */
    a = 1. / 120.;
    a *= x;
    a += 4.1666666666666666666666666666666666666666666666666e-2;
    a *= x;
    a += 0.166666666666666666666666666666666666666666666666665;
    a *= x;
    a += 0.5;
    a *= x;
    a += 1.0;
    a *= x;
    a += 1.0;

    /* Build 2^n in double. */
    u.d = 0;
    n += 1023;
    u.s[3] = (unsigned short)((n << 4) & 0x7FF0);

    return a * u.d;
}
