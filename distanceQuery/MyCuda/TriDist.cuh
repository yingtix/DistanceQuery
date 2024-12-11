#pragma once

#define VmV(T, Q, P) \
    {                \
        T = Q - P;   \
    }
#define VpVxS(Vr, V1, V2, s) \
    {                        \
        Vr = V1 + V2 * s;    \
    }
#define VdotV(A, B) dot(A, B)
#define VcV(A, B) \
    {             \
        A = B;    \
    }
#define VpV(T, Q, P) \
    {                \
        T = Q + P;   \
    }
#define VcrossV(A, B, C) \
    {                    \
        A = cross(B, C); \
    }
#define VxS(A, B, s) \
    {                \
        A = B * s;   \
    }
#define VdistV2(A, B) norm2(A - B)

//--------------------------------------------------------------------------
// SegPoints()
//
// Returns closest points between an segment pair.
// Implemented from an algorithm described in
//
// Vladimir J. Lumelsky,
// On fast computation of distance between line segments.
// In Information Processing Letters, no. 21, pages 55-61, 1985.
//--------------------------------------------------------------------------
__host__ __device__ void
SegPoints(float3 &VEC,
          float3 &X, float3 &Y,             // closest points
          const float3 &P, const float3 &A, // seg 1 origin, vector
          const float3 &Q, const float3 &B) // seg 2 origin, vector
{
    float3 T, TMP;
    float A_dot_A, B_dot_B, A_dot_B, A_dot_T, B_dot_T;

    VmV(T, Q, P);
    A_dot_A = VdotV(A, A);
    B_dot_B = VdotV(B, B);
    A_dot_B = VdotV(A, B);
    A_dot_T = VdotV(A, T);
    B_dot_T = VdotV(B, T);

    // t parameterizes ray P,A
    // u parameterizes ray Q,B

    double t, u;

    // compute t for the closest point on ray P,A to
    // ray Q,B

    double denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B;

    t = (A_dot_T * B_dot_B - B_dot_T * A_dot_B) / denom;

    // clamp result so t is on the segment P,A

    if ((t < 0) || isnan(t))
        t = 0;
    else if (t > 1)
        t = 1;

    // find u for point on ray Q,B closest to point at t

    u = (t * A_dot_B - B_dot_T) / B_dot_B;

    // if u is on segment Q,B, t and u correspond to
    // closest points, otherwise, clamp u, recompute and
    // clamp t

    if ((u <= 0) || isnan(u))
    {

        VcV(Y, Q);

        t = A_dot_T / A_dot_A;

        if ((t <= 0) || isnan(t))
        {
            VcV(X, P);
            VmV(VEC, Q, P);
        }
        else if (t >= 1)
        {
            VpV(X, P, A);
            VmV(VEC, Q, X);
        }
        else
        {
            VpVxS(X, P, A, t);
            VcrossV(TMP, T, A);
            VcrossV(VEC, A, TMP);
        }
    }
    else if (u >= 1)
    {

        VpV(Y, Q, B);

        t = (A_dot_B + A_dot_T) / A_dot_A;

        if ((t <= 0) || isnan(t))
        {
            VcV(X, P);
            VmV(VEC, Y, P);
        }
        else if (t >= 1)
        {
            VpV(X, P, A);
            VmV(VEC, Y, X);
        }
        else
        {
            VpVxS(X, P, A, t);
            VmV(T, Y, P);
            VcrossV(TMP, T, A);
            VcrossV(VEC, A, TMP);
        }
    }
    else
    {

        VpVxS(Y, Q, B, u);

        if ((t <= 0) || isnan(t))
        {
            VcV(X, P);
            VcrossV(TMP, T, B);
            VcrossV(VEC, B, TMP);
        }
        else if (t >= 1)
        {
            VpV(X, P, A);
            VmV(T, Q, X);
            VcrossV(TMP, T, B);
            VcrossV(VEC, B, TMP);
        }
        else
        {
            VpVxS(X, P, A, t);
            VcrossV(VEC, A, B);
            if (VdotV(VEC, T) < 0)
            {
                VxS(VEC, VEC, -1);
            }
        }
    }
}

//--------------------------------------------------------------------------
// TriDist()
//
// Computes the closest points on two triangles, and returns the
// distance between them.
//
// S and T are the triangles, stored tri[point][dimension].
//
// If the triangles are disjoint, P and Q give the closest points of
// S and T respectively. However, if the triangles overlap, P and Q
// are basically a random pair of points from the triangles, not
// coincident points on the intersection of the triangles, as might
// be expected.
//--------------------------------------------------------------------------

__host__ __device__ float
TriDistPQP(float3 &P, float3 &Q, const float3 S[3], const float3 T[3])
{
    // Compute vectors along the 6 sides

    float3 Sv[3], Tv[3], VEC;

    Sv[0] = S[1] - S[0];
    Sv[1] = S[2] - S[1];
    Sv[2] = S[0] - S[2];

    Tv[0] = T[1] - T[0];
    Tv[1] = T[2] - T[1];
    Tv[2] = T[0] - T[2];

    // For each edge pair, the vector connecting the closest points
    // of the edges defines a slab (parallel planes at head and tail
    // enclose the slab). If we can show that the off-edge vertex of
    // each triangle is outside of the slab, then the closest points
    // of the edges are the closest points for the triangles.
    // Even if these tests fail, it may be helpful to know the closest
    // points found, and whether the triangles were shown disjoint

    float3 V, Z, minP, minQ;
    float mindd;
    int shown_disjoint = 0;

    mindd = VdistV2(S[0], T[0]) + 1; // Set first minimum safely high

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // Find closest points on edges i & j, plus the
            // vector (and distance squared) between these points

            SegPoints(VEC, P, Q, S[i], Sv[i], T[j], Tv[j]);

            VmV(V, Q, P);
            double dd = VdotV(V, V);

            // Verify this closest point pair only if the distance
            // squared is less than the minimum found thus far.

            if (dd <= mindd)
            {
                VcV(minP, P);
                VcV(minQ, Q);
                mindd = dd;

                VmV(Z, S[(i + 2) % 3], P);
                double a = VdotV(Z, VEC);
                VmV(Z, T[(j + 2) % 3], Q);
                double b = VdotV(Z, VEC);

                if ((a <= 0) && (b >= 0))
                    // return sqrt(dd);
                    return dd;

                double p = VdotV(V, VEC);

                if (a < 0)
                    a = 0;
                if (b > 0)
                    b = 0;
                if ((p - a + b) > 0)
                    shown_disjoint = 1;
            }
        }
    }

    // No edge pairs contained the closest points.
    // either:
    // 1. one of the closest points is a vertex, and the
    //    other point is interior to a face.
    // 2. the triangles are overlapping.
    // 3. an edge of one triangle is parallel to the other's face. If
    //    cases 1 and 2 are not true, then the closest points from the 9
    //    edge pairs checks above can be taken as closest points for the
    //    triangles.
    // 4. possibly, the triangles were degenerate.  When the
    //    triangle points are nearly colinear or coincident, one
    //    of above tests might fail even though the edges tested
    //    contain the closest points.

    // First check for case 1
    float3 Sn;
    float Snl;
    VcrossV(Sn, Sv[0], Sv[1]); // Compute normal to S triangle
    Snl = VdotV(Sn, Sn);       // Compute square of length of normal

    // If cross product is long enough,

    if (Snl > 1e-15)
    {
        // Get projection lengths of T points

        float Tp[3];

        VmV(V, S[0], T[0]);
        Tp[0] = VdotV(V, Sn);

        VmV(V, S[0], T[1]);
        Tp[1] = VdotV(V, Sn);

        VmV(V, S[0], T[2]);
        Tp[2] = VdotV(V, Sn);

        // If Sn is a separating direction,
        // find point with smallest projection

        int point = -1;
        if ((Tp[0] > 0) && (Tp[1] > 0) && (Tp[2] > 0))
        {
            if (Tp[0] < Tp[1])
                point = 0;
            else
                point = 1;
            if (Tp[2] < Tp[point])
                point = 2;
        }
        else if ((Tp[0] < 0) && (Tp[1] < 0) && (Tp[2] < 0))
        {
            if (Tp[0] > Tp[1])
                point = 0;
            else
                point = 1;
            if (Tp[2] > Tp[point])
                point = 2;
        }

        // If Sn is a separating direction,

        if (point >= 0)
        {
            shown_disjoint = 1;

            // Test whether the point found, when projected onto the
            // other triangle, lies within the face.

            VmV(V, T[point], S[0]);
            VcrossV(Z, Sn, Sv[0]);
            if (VdotV(V, Z) > 0)
            {
                VmV(V, T[point], S[1]);
                VcrossV(Z, Sn, Sv[1]);
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, T[point], S[2]);
                    VcrossV(Z, Sn, Sv[2]);
                    if (VdotV(V, Z) > 0)
                    {
                        // T[point] passed the test - it's a closest point for
                        // the T triangle; the other point is on the face of S

                        VpVxS(P, T[point], Sn, Tp[point] / Snl);
                        VcV(Q, T[point]);
                        // return sqrt(VdistV2(P, Q));
                        return VdistV2(P, Q);
                    }
                }
            }
        }
    }

    float3 Tn;
    float Tnl;
    VcrossV(Tn, Tv[0], Tv[1]);
    Tnl = VdotV(Tn, Tn);

    if (Tnl > 1e-15)
    {
        float Sp[3];

        VmV(V, T[0], S[0]);
        Sp[0] = VdotV(V, Tn);

        VmV(V, T[0], S[1]);
        Sp[1] = VdotV(V, Tn);

        VmV(V, T[0], S[2]);
        Sp[2] = VdotV(V, Tn);

        int point = -1;
        if ((Sp[0] > 0) && (Sp[1] > 0) && (Sp[2] > 0))
        {
            if (Sp[0] < Sp[1])
                point = 0;
            else
                point = 1;
            if (Sp[2] < Sp[point])
                point = 2;
        }
        else if ((Sp[0] < 0) && (Sp[1] < 0) && (Sp[2] < 0))
        {
            if (Sp[0] > Sp[1])
                point = 0;
            else
                point = 1;
            if (Sp[2] > Sp[point])
                point = 2;
        }

        if (point >= 0)
        {
            shown_disjoint = 1;

            VmV(V, S[point], T[0]);
            VcrossV(Z, Tn, Tv[0]);
            if (VdotV(V, Z) > 0)
            {
                VmV(V, S[point], T[1]);
                VcrossV(Z, Tn, Tv[1]);
                if (VdotV(V, Z) > 0)
                {
                    VmV(V, S[point], T[2]);
                    VcrossV(Z, Tn, Tv[2]);
                    if (VdotV(V, Z) > 0)
                    {
                        VcV(P, S[point]);
                        VpVxS(Q, S[point], Tn, Sp[point] / Tnl);
                        // return sqrt(VdistV2(P, Q));
                        return VdistV2(P, Q);
                    }
                }
            }
        }
    }

    // Case 1 can't be shown.
    // If one of these tests showed the triangles disjoint,
    // we assume case 3 or 4, otherwise we conclude case 2,
    // that the triangles overlap.

    if (shown_disjoint)
    {
        VcV(P, minP);
        VcV(Q, minQ);
        // return sqrt(mindd);
        return mindd;
    }
    else
        return 0;
}

__host__ __device__ float
triDist(
    const float3 &P1, const float3 &P2, const float3 &P3,
    const float3 &Q1, const float3 &Q2, const float3 &Q3,
    float3 &rP, float3 &rQ)
{
    float3 S[3];
    float3 T[3];

    S[0] = P1, S[1] = P2, S[2] = P3;
    T[0] = Q1, T[1] = Q2, T[2] = Q3;

    return TriDistPQP(rP, rQ, S, T);
}