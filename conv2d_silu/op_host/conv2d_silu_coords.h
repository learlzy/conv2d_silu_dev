#ifndef CONV2D_SILU_COORDS_H
#define CONV2D_SILU_COORDS_H


template <
    int RANK_,                 ///< Logical rank of coordinate
    class Index_ = uint32_t,   ///< Index type used for each dimension
    class LongIndex_ = int64_t ///< Long index type used for linear offsets
    >
struct Coord {
public:
    // Number of elements in Coord
    static const int RANK = RANK_;

    // Index typen used to store elements
    using Index = Index_;

    // Type used to represent linear offsets
    using LongIndex = LongIndex_;

    // Default ctor initializes uniformly
    constexpr explicit Coord(Index value = Index(0)) : idx{}
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    // Constructs from an array of integers
    constexpr Coord(Index const (&idx_)[RANK]) : idx{}
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    
    constexpr int Argmin() const
    {
        return ArgminImpl<1>(0);
    }

    // Returns the index of the dimension with greatest value
    
    constexpr int Argmax() const
    {
        return ArgmaxImpl<1>(0);
    }

    // Returns true if Coord is non-zero
    
    constexpr explicit operator bool() const
    {
        return AnyImpl<0>();
    }

    // Return true if Coord is uniformly zero.
    
    constexpr bool operator!() const
    {
        return !AnyImpl<0>();
    }

    // Element-wise addition
    
    constexpr Coord operator+(Coord const& b) const
    {
        Coord c;
        AddCoordImpl<0>(c, b);
        return c;
    }

    // Add a scalar to each element
    
    constexpr Coord operator+(const Index val) const
    {
        Coord c;
        AddScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise subtraction
    
    constexpr Coord operator-(Coord const& b) const
    {
        Coord c;
        SubCoordImpl<0>(c, b);
        return c;
    }

    // Subtract a scalar from each element
    
    constexpr Coord operator-(Index const val) const
    {
        Coord c;
        SubScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise multiply
    
    constexpr Coord operator*(Coord const& b) const
    {
        Coord c;
        MulCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise division
    
    constexpr Coord operator/(Coord const& b) const
    {
        Coord c;
        DivCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise mod
    
    constexpr Coord operator%(Coord const& b) const
    {
        Coord c;
        ModCoordImpl<0>(c, b);
        return c;
    }

    // In-place addition
    
    constexpr Coord& operator+=(Coord const& b)
    {
        PlusEqualImpl<0>(b);
        return *this;
    }

    // In-place equal
    
    constexpr bool operator==(Coord const& b) const
    {
        return EqualCoordImpl<0>(b);
    }

    // In-place equal
    
    constexpr bool operator==(Index const val) const
    {
        return EqualScalarImpl<0>(val);
    }

    // Member acces operator
    constexpr Index& operator[](int dim)
    {
        return idx[dim];
    }

    // Member access operator
    constexpr Index const& operator[](int dim) const
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    Index& At()
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    
    Index& At(int dim)
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    Index const& At() const
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    
    Index const& At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    auto GetCoordByAxis() const
    {
        Index idx_[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{idx_};
    }

    
    static constexpr Coord Min(Coord const& a, Coord const& b)
    {
        Coord res;
        for (int i = 0; i < RANK; ++i) {
            res[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return res;
    }

private:
    template <int N>
    constexpr int ArgminImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        } else {
            return ArgminImpl<N + 1>(idx[N] < idx[i] ? N : i);
        }
    }

    template <int N>
    constexpr int ArgmaxImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        } else {
            return ArgmaxImpl<N + 1>(idx[N] > idx[i] ? N : i);
        }
    }

    template <int N>
    constexpr bool AnyImpl() const
    {
        if constexpr (N == RANK) {
            return false;
        } else {
            return idx[N] || AnyImpl<N + 1>();
        }
    }

    template <int N>
    constexpr void AddCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + b.idx[N];
            AddCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    constexpr void AddScalarImpl(Coord& c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + val;
            AddScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    constexpr void SubCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - b.idx[N];
            SubCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    constexpr void SubScalarImpl(Coord& c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - val;
            SubScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    constexpr void MulCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] * b.idx[N];
            MulCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    constexpr void DivCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] / b.idx[N];
            DivCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    constexpr void ModCoordImpl(Coord& c, Coord const& b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] % b.idx[N];
            ModCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    constexpr void PlusEqualImpl(Coord const& b)
    {
        if constexpr (N < RANK) {
            idx[N] += b.idx[N];
            PlusEqualImpl<N + 1>(b);
        }
    }

    template <int N>
    constexpr bool EqualCoordImpl(Coord const& b) const
    {
        if constexpr (N == RANK) {
            return true;
        } else {
            return idx[N] == b.idx[N] && EqualCoordImpl<N + 1>(b);
        }
    }

    template <int N>
    constexpr bool EqualScalarImpl(Index const val) const
    {
        if constexpr (N == RANK) {
            return true;
        } else {
            return idx[N] == val && EqualScalarImpl<N + 1>(val);
        }
    }

    // Indices
    Index idx[RANK];
};

// Helper to make a 1-element coordinate
template <class T>
constexpr Coord<1, T> MakeCoord(T dim0)
{
    T values[1] = {dim0};
    return Coord<1, T>(values);
}

/// Helper to make a 2-element coordinate
template <class T>
constexpr Coord<2, T> MakeCoord(T dim0, T dim1)
{
    T values[2] = {dim0, dim1};
    return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <class T>
constexpr Coord<3, T> MakeCoord(T dim0, T dim1, T dim2)
{
    T values[3] = {dim0, dim1, dim2};
    return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <class T>
constexpr Coord<4, T> MakeCoord(T dim0, T dim1, T dim2, T dim3)
{
    T values[4] = {dim0, dim1, dim2, dim3};
    return Coord<4, T>(values);
}

/// Helper to make a 5-element coordinate
template <class T>
constexpr Coord<5, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4)
{
    T values[5] = {dim0, dim1, dim2, dim3, dim4};
    return Coord<5, T>(values);
}

/// Helper to make a 6-element coordinate
template <class T>
constexpr Coord<6, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5)
{
    T values[6] = {dim0, dim1, dim2, dim3, dim4, dim5};
    return Coord<6, T>(values);
}

/// Helper to make a 7-element coordinate
template <class T>
constexpr Coord<7, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5, T dim6)
{
    T values[7] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6};
    return Coord<7, T>(values);
}


template <
    uint32_t Ho_ = 1, uint32_t Wo_ = 1,
    uint32_t Cin1_ = 1>
struct Conv2dFmapL1Shape { // (Ho, Wo, Cin1)
    static constexpr uint32_t Ho = Ho_;
    static constexpr uint32_t Wo = Wo_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    
    static Coord<3> ToCoord()
    {
        return MakeCoord(Ho, Wo, Cin1);
    }
};

template <
    uint32_t Cout_ = 16,
    uint32_t Cin1_ = 1>
struct Conv2dFilterL1Shape { // (Cout, Cin1)
    static constexpr uint32_t Cout = Cout_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    
    static Coord<2> ToCoord()
    {
        return MakeCoord(Cout, Cin1);
    }
};

template <uint32_t M_ = 16, uint32_t N_ = 16, uint32_t K_ = 16>
struct Conv2dL0Shape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    /// Returns a Coord object
    
    static Coord<3> ToCoord()
    {
        return MakeCoord(M, N, K);
    }
};

struct Conv2dFmapCoord : public Coord<5, uint32_t> { // (Batch, C1, H, W, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t C1_INDEX = 1;
    static constexpr uint32_t H_INDEX = 2;
    static constexpr uint32_t W_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    
    Conv2dFmapCoord()
    {}

    /// Constructs from Coord<5>
    
    Conv2dFmapCoord(Coord<5, Index> const& coord) : Base(coord)
    {}

    /// Helper to construct from C1, H, W, C0
    
    Conv2dFmapCoord(Index batch, Index c1, Index h, Index w, Index c0) : Base(MakeCoord(batch, c1, h, w, c0))
    {}

    
    Conv2dFmapCoord(LongIndex batch, LongIndex c1, LongIndex h, LongIndex w, LongIndex c0)
        : Base(MakeCoord(Index(batch), Index(c1), Index(h), Index(w), Index(c0)))
    {}

    
    Index const& c1() const
    {
        return this->At(C1_INDEX);
    }
    
    Index& c1()
    {
        return this->At(C1_INDEX);
    }

    
    Index const& batch() const
    {
        return this->At(BATCH_INDEX);
    }
    
    Index& batch()
    {
        return this->At(BATCH_INDEX);
    }

    
    Index const& h() const
    {
        return this->At(H_INDEX);
    }
    
    Index& h()
    {
        return this->At(H_INDEX);
    }

    
    Index const& w() const
    {
        return this->At(W_INDEX);
    }
    
    Index& w()
    {
        return this->At(W_INDEX);
    }

    
    Index const& c0() const
    {
        return this->At(C0_INDEX);
    }
    
    Index& c0()
    {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    
    Conv2dFmapCoord operator+(Base const& b) const
    {
        return Conv2dFmapCoord(Base::operator+(b));
    }

    /// In-place addition
    
    Conv2dFmapCoord& operator+=(Base const& b)
    {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dFilterCoord : public Coord<5, uint32_t> { // (Cin1, Kh, Kw, Cout, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t CIN1_INDEX = 0;
    static constexpr uint32_t KH_INDEX = 1;
    static constexpr uint32_t KW_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    
    Conv2dFilterCoord()
    {}

    /// Constructs from Coord<5>
    
    Conv2dFilterCoord(Coord<5, Index> const& coord) : Base(coord)
    {}

    /// Helper to construct from Cin1, Kh, Kw, Cout, C0
    
    Conv2dFilterCoord(Index cin1, Index kh, Index kw, Index cout, Index c0) : Base(MakeCoord(cin1, kh, kw, cout, c0))
    {}

    
    Conv2dFilterCoord(LongIndex cin1, LongIndex kh, LongIndex kw, LongIndex cout, LongIndex c0)
        : Base(MakeCoord(Index(cin1), Index(kh), Index(kw), Index(cout), Index(c0)))
    {}

    
    Index const& cin1() const
    {
        return this->At(CIN1_INDEX);
    }
    
    Index& cin1()
    {
        return this->At(CIN1_INDEX);
    }

    
    Index const& kh() const
    {
        return this->At(KH_INDEX);
    }
    
    Index& kh()
    {
        return this->At(KH_INDEX);
    }

    
    Index const& kw() const
    {
        return this->At(KW_INDEX);
    }
    
    Index& kw()
    {
        return this->At(KW_INDEX);
    }

    
    Index const& cout() const
    {
        return this->At(COUT_INDEX);
    }
    
    Index& cout()
    {
        return this->At(COUT_INDEX);
    }

    
    Index const& c0() const
    {
        return this->At(C0_INDEX);
    }
    
    Index& c0()
    {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    
    Conv2dFilterCoord operator+(Base const& b) const
    {
        return Conv2dFilterCoord(Base::operator+(b));
    }

    /// In-place addition
    
    Conv2dFilterCoord& operator+=(Base const& b)
    {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dHoWoCoCoord : public Coord<3, uint32_t> { // (Ho, Wo, Cout)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t HO_INDEX = 0;
    static constexpr uint32_t WO_INDEX = 1;
    static constexpr uint32_t COUT_INDEX = 2;

    /// Default ctor
    
    Conv2dHoWoCoCoord()
    {}

    /// Constructs from Coord<3>
    
    Conv2dHoWoCoCoord(Coord<3, Index> const& coord) : Base(coord)
    {}

    /// Helper to construct from Ho, Wo, Cout
    
    Conv2dHoWoCoCoord(Index ho, Index wo, Index cout) : Base(MakeCoord(ho, wo, cout))
    {}

    
    Conv2dHoWoCoCoord(LongIndex ho, LongIndex wo, LongIndex cout) : Base(MakeCoord(Index(ho), Index(wo), Index(cout)))
    {}

    
    Index const& ho() const
    {
        return this->At(HO_INDEX);
    }

    
    Index& ho()
    {
        return this->At(HO_INDEX);
    }

    
    Index const& wo() const
    {
        return this->At(WO_INDEX);
    }

    
    Index& wo()
    {
        return this->At(WO_INDEX);
    }

    
    Index howo() const
    {
        return this->At(HO_INDEX) * this->At(WO_INDEX);
    }

    
    Index const& cout() const
    {
        return this->At(COUT_INDEX);
    }

    
    Index& cout()
    {
        return this->At(COUT_INDEX);
    }

    /// Element-wise addition
    
    Conv2dHoWoCoCoord operator+(Base const& b) const
    {
        return Conv2dHoWoCoCoord(Base::operator+(b));
    }

    /// In-place addition
    
    Conv2dHoWoCoCoord& operator+=(Base const& b)
    {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dCoord : public Coord<5, uint32_t> { // (Batch, H, W, Cout, Cin1)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t H_INDEX = 1;
    static constexpr uint32_t W_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t CIN1_INDEX = 4;

    /// Default ctor
    
    Conv2dCoord()
    {}

    /// Constructs from Coord<5>
    
    Conv2dCoord(Coord<5, Index> const& coord) : Base(coord)
    {}

    /// Helper to construct from Batch, H, W, Cout, Cin1
    
    Conv2dCoord(Index batch, Index h, Index w, Index cout, Index cin1) : Base(MakeCoord(batch, h, w, cout, cin1))
    {}

    
    Conv2dCoord(LongIndex batch, LongIndex h, LongIndex w, LongIndex cout, LongIndex cin1)
        : Base(MakeCoord(Index(batch), Index(h), Index(w), Index(cout), Index(cin1)))
    {}

    
    Index const& batch() const
    {
        return this->At(BATCH_INDEX);
    }
    
    Index& batch()
    {
        return this->At(BATCH_INDEX);
    }

    
    Index const& h() const
    {
        return this->At(H_INDEX);
    }
    
    Index& h()
    {
        return this->At(H_INDEX);
    }

    
    Index const& w() const
    {
        return this->At(W_INDEX);
    }
    
    Index& w()
    {
        return this->At(W_INDEX);
    }

    
    Index const& cout() const
    {
        return this->At(COUT_INDEX);
    }
    
    Index& cout()
    {
        return this->At(COUT_INDEX);
    }

    
    Index const& cin1() const
    {
        return this->At(CIN1_INDEX);
    }
    
    Index& cin1()
    {
        return this->At(CIN1_INDEX);
    }

    /// Element-wise addition
    
    Conv2dCoord operator+(Base const& b) const
    {
        return Conv2dCoord(Base::operator+(b));
    }

    /// In-place addition
    
    Conv2dCoord& operator+=(Base const& b)
    {
        Base::operator+=(b);
        return *this;
    }

    
    auto GetHoWoCoCoord() const
    {
        return this->GetCoordByAxis<H_INDEX, W_INDEX, COUT_INDEX>();
    }
};

class Conv2dFilterParams {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16;
    using Ks = Coord<2, ShortIndex>;
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;

private:
    Ks ks;
    Pads pads;
    Strides strides;
    Dilations dilations;

public:
    Conv2dFilterParams(
        ShortIndex kh = 0, ShortIndex kw = 0, ShortIndex padLeft = 0, ShortIndex padRight = 0, ShortIndex padTop = 0,
        ShortIndex padBottom = 0, ShortIndex strideH = 0, ShortIndex strideW = 0, ShortIndex dilationH = 0,
        ShortIndex dilationW = 0)
        : ks(MakeCoord(kh, kw)),
          pads(MakeCoord(padLeft, padRight, padTop, padBottom)),
          strides(MakeCoord(strideH, strideW)),
          dilations(MakeCoord(dilationH, dilationW))
    {}

    
    ShortIndex const& kh() const
    {
        return this->ks[0];
    }
    
    ShortIndex& kh()
    {
        return this->ks[0];
    }

    
    ShortIndex const& kw() const
    {
        return this->ks[1];
    }
    
    ShortIndex& kw()
    {
        return this->ks[1];
    }

    
    ShortIndex const& padLeft() const
    {
        return this->pads[0];
    }
    
    ShortIndex const& padLeft()
    {
        return this->pads[0];
    }

    
    ShortIndex const& padRight() const
    {
        return this->pads[1];
    }
    
    ShortIndex const& padRight()
    {
        return this->pads[1];
    }

    
    ShortIndex const& padTop() const
    {
        return this->pads[2];
    }
    
    ShortIndex const& padTop()
    {
        return this->pads[2];
    }

    
    ShortIndex const& padBottom() const
    {
        return this->pads[3];
    }
    
    ShortIndex const& padBottom()
    {
        return this->pads[3];
    }

    
    ShortIndex const& strideH() const
    {
        return this->strides[0];
    }
    
    ShortIndex const& strideH()
    {
        return this->strides[0];
    }

    
    ShortIndex const& strideW() const
    {
        return this->strides[1];
    }
    
    ShortIndex const& strideW()
    {
        return this->strides[1];
    }

    
    ShortIndex const& dilationH() const
    {
        return this->dilations[0];
    }
    
    ShortIndex const& dilationH()
    {
        return this->dilations[0];
    }

    
    ShortIndex const& dilationW() const
    {
        return this->dilations[1];
    }
    
    ShortIndex const& dilationW()
    {
        return this->dilations[1];
    }
};

struct Conv2dParams {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16; // todo:根据具体数据类型来赋值
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;

private:
    // Batch, Hi, Wi, Cin, Cout, Kh, Kw
    Conv2dFmapCoord fmapShape;     // {Batch, Cin1, Hi, Wi, C0}
    Conv2dFilterCoord filterShape; // {Cin1, Kh, Kw, Cout, C0}
    Conv2dFmapCoord outputShape;   // {Batch, Cout1, Ho, Wo, C0}
    Conv2dFilterParams configs;    // {Ks, Pads, Strides, Dilations}
    Conv2dCoord postIm2colShape;   // {Batch, Ho, Wo, Cout, Cin1}
public:
    /// Default ctor
    
    Conv2dParams()
    {}

    
    Conv2dParams(
        Index batch, Index hi, Index wi, Index cin, Index cout, ShortIndex kh, ShortIndex kw, ShortIndex padLeft,
        ShortIndex padRight, ShortIndex padTop, ShortIndex padBottom, ShortIndex strideH, ShortIndex strideW,
        ShortIndex dilationH, ShortIndex dilationW)
        : fmapShape(MakeCoord(batch, CeilDiv(cin, C0), hi, wi, C0)),
          filterShape(MakeCoord(CeilDiv(cin, C0), (Index)kh, (Index)kw, cout, C0)),
          configs(kh, kw, padLeft, padRight, padTop, padBottom, strideH, strideW, dilationH, dilationW)
    {
        Index cout1 = CeilDiv(cout, C0);
        Index ho = (hi + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
        Index wo = (wi + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
        outputShape = MakeCoord(batch, cout1, ho, wo, C0);
        postIm2colShape = MakeCoord(batch, ho, wo, cout, filterShape.cin1());
    }

    
    static Conv2dParams MakeConv2dParams(
        const Index* dataSizes,        // [Batch, Hi, Wi, Cin, Cout]
        const ShortIndex* filterSizes, // [Kh, Kw]
        const ShortIndex* pads,        // [padLeft, padRight, padTop, padBottom]
        const ShortIndex* strides,     // [strideH, strideW]
        const ShortIndex* dilations)   // [dilationH, dilationW]
    {
        return Conv2dParams(
            dataSizes[0], dataSizes[1], dataSizes[2], dataSizes[3], dataSizes[4], filterSizes[0], filterSizes[1],
            pads[0], pads[1], pads[2], pads[3], strides[0], strides[1], strides[2], strides[3]);
    }

    
    Conv2dFilterParams const& getFilterParams() const
    {
        return this->configs;
    }

    
    Conv2dFmapCoord const& getOutputShape() const
    {
        return this->outputShape;
    }

    
    Conv2dCoord const& getPostIm2colShape() const
    {
        return this->postIm2colShape;
    }

    
    Index const& batch() const
    {
        return this->fmapShape.batch();
    }
    
    Index& batch()
    {
        return this->fmapShape.batch();
    }

    
    Index const& hi() const
    {
        return this->fmapShape.h();
    }
    
    Index& hi()
    {
        return this->fmapShape.h();
    }

    
    Index const& wi() const
    {
        return this->fmapShape.w();
    }
    
    Index& wi()
    {
        return this->fmapShape.w();
    }

    
    Index cin() const
    {
        return this->fmapShape.c1() * C0;
    }

    
    Index const& cin1() const
    {
        return this->fmapShape.c1();
    }
    
    Index& cin1()
    {
        return this->fmapShape.c1();
    }

    
    Index const& cout() const
    {
        return this->filterShape.cout();
    }
    
    Index& cout()
    {
        return this->filterShape.cout();
    }

    
    Index const& cout1() const
    {
        return this->outputShape.c1();
    }
    
    Index& cout1()
    {
        return this->outputShape.c1();
    }

    
    Index coutRound() const
    {
        return (this->filterShape.cout() + C0 - 1) / C0 * C0;
    }

    
    Index const& ho() const
    {
        return this->outputShape.h();
    }
    
    Index& ho()
    {
        return this->outputShape.h();
    }

    
    Index const& wo() const
    {
        return this->outputShape.w();
    }
    
    Index& wo()
    {
        return this->outputShape.w();
    }

    
    Index howo() const
    {
        return this->outputShape.h() * this->outputShape.w();
    }

    
    Index howoRound() const
    {
        return (this->howo() + C0 - 1) / C0 * C0;
    }

    
    ShortIndex const& kh() const
    {
        return this->configs.kh();
    }
    
    ShortIndex& kh()
    {
        return this->configs.kh();
    }

    
    ShortIndex const& kw() const
    {
        return this->configs.kw();
    }
    
    ShortIndex& kw()
    {
        return this->configs.kw();
    }

    
    ShortIndex const& padTop() const
    {
        return this->configs.padTop();
    }
    
    ShortIndex const& padTop()
    {
        return this->configs.padTop();
    }

    
    ShortIndex const& padBottom() const
    {
        return this->configs.padBottom();
    }
    
    ShortIndex const& padBottom()
    {
        return this->configs.padBottom();
    }

    
    ShortIndex const& padLeft() const
    {
        return this->configs.padLeft();
    }
    
    ShortIndex const& padLeft()
    {
        return this->configs.padLeft();
    }

    
    ShortIndex const& padRight() const
    {
        return this->configs.padRight();
    }
    
    ShortIndex const& padRight()
    {
        return this->configs.padRight();
    }

    
    ShortIndex const& strideH() const
    {
        return this->configs.strideH();
    }
    
    ShortIndex const& strideH()
    {
        return this->configs.strideH();
    }

    
    ShortIndex const& strideW() const
    {
        return this->configs.strideW();
    }
    
    ShortIndex const& strideW()
    {
        return this->configs.strideW();
    }

    
    ShortIndex const& dilationH() const
    {
        return this->configs.dilationH();
    }
    
    ShortIndex const& dilationH()
    {
        return this->configs.dilationH();
    }

    
    ShortIndex const& dilationW() const
    {
        return this->configs.dilationW();
    }
    
    ShortIndex const& dilationW()
    {
        return this->configs.dilationW();
    }
};

#endif // CATLASS_COORD_HPP