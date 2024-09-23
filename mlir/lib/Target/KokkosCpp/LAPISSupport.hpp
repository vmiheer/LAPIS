#include <Kokkos_Core.hpp>
#include <type_traits>
#include <cstdint>
#include <unistd.h>

// If building a CPP driver, we can use the original StridedMemRefType class from MLIR,
// so do not redefine it here.
#ifndef PYTACO_CPP_DRIVER
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
#endif

// If building a CPP driver, need to provide a version of
// _mlir_ciface_newSparseTensor() that takes underlying integer types, not enum types like DimLevelType.
// The MLIR-Kokkos generated code doesn't know about the enum types at all.
#ifdef PYTACO_CPP_DRIVER
int8_t* _mlir_ciface_newSparseTensor(
  StridedMemRefType<index_type, 1> *dimSizesRef,
  StridedMemRefType<index_type, 1> *lvlSizesRef,
  StridedMemRefType<int8_t, 1> *lvlTypesRef,
  StridedMemRefType<index_type, 1> *lvl2dimRef,
  StridedMemRefType<index_type, 1> *dim2lvlRef, int ptrTp,
  int indTp, int valTp, int action, int8_t* ptr) {
    return (int8_t*) _mlir_ciface_newSparseTensor(dimSizesRef, lvlSizesRef,
      reinterpret_cast<StridedMemRefType<DimLevelType, 1>*>(lvlTypesRef),
      lvl2dimRef, dim2lvlRef, (OverheadType) ptrTp, (OverheadType) indTp,
      (PrimaryType) valTp, (Action) action, ptr);
  }
#endif

namespace LAPIS
{
  using TeamPolicy = Kokkos::TeamPolicy<>;
  using TeamMember = typename TeamPol::member_type;

  template<typename V>
    StridedMemRefType<typename V::value_type, V::rank> viewToStridedMemref(const V& v)
    {
      StridedMemRefType<typename V::value_type, V::rank> smr;
      smr.basePtr = v.data();
      smr.data = v.data();
      smr.offset = 0;
      for(int i = 0; i < int(V::rank); i++)
      {
        smr.sizes[i] = v.extent(i);
        smr.strides[i] = v.stride(i);
      }
      return smr;
    }

  template<typename V>
    V stridedMemrefToView(const StridedMemRefType<typename V::value_type, V::rank>& smr)
    {
      using Layout = typename V::array_layout;
      static_assert(std::is_same_v<typename V::memory_space, Kokkos::HostSpace> ||
          std::is_same_v<typename V::memory_space, Kokkos::AnonymousSpace>,
          "Can only convert a StridedMemRefType to a Kokkos::View in HostSpace.");
      if constexpr(std::is_same_v<Layout, Kokkos::LayoutStride>)
      {
        Layout layout(
            (0 < V::rank) ? smr.sizes[0] : 0U,
            (0 < V::rank) ? smr.strides[0] : 0U,
            (1 < V::rank) ? smr.sizes[1] : 0U,
            (1 < V::rank) ? smr.strides[1] : 0U,
            (2 < V::rank) ? smr.sizes[2] : 0U,
            (2 < V::rank) ? smr.strides[2] : 0U,
            (3 < V::rank) ? smr.sizes[3] : 0U,
            (3 < V::rank) ? smr.strides[3] : 0U,
            (4 < V::rank) ? smr.sizes[4] : 0U,
            (4 < V::rank) ? smr.strides[4] : 0U,
            (5 < V::rank) ? smr.sizes[5] : 0U,
            (5 < V::rank) ? smr.strides[5] : 0U,
            (6 < V::rank) ? smr.sizes[6] : 0U,
            (6 < V::rank) ? smr.strides[6] : 0U,
            (7 < V::rank) ? smr.sizes[7] : 0U,
            (7 < V::rank) ? smr.strides[7] : 0U);
        return V(&smr.data[smr.offset], layout);
      }
      Layout layout(
          (0 < V::rank) ? smr.sizes[0] : 0U,
          (1 < V::rank) ? smr.sizes[1] : 0U,
          (2 < V::rank) ? smr.sizes[2] : 0U,
          (3 < V::rank) ? smr.sizes[3] : 0U,
          (4 < V::rank) ? smr.sizes[4] : 0U,
          (5 < V::rank) ? smr.sizes[5] : 0U,
          (6 < V::rank) ? smr.sizes[6] : 0U,
          (7 < V::rank) ? smr.sizes[7] : 0U);
      if constexpr(std::is_same_v<Layout, Kokkos::LayoutLeft>)
      {
        int64_t expectedStride = 1;
        for(int i = 0; i < int(V::rank); i++)
        {
          if(expectedStride != smr.strides[i])
            Kokkos::abort("Cannot convert non-contiguous StridedMemRefType to LayoutLeft Kokkos::View");
          expectedStride *= smr.sizes[i];
        }
      }
      else if constexpr(std::is_same_v<Layout, Kokkos::LayoutRight>)
      {
        int64_t expectedStride = 1;
        for(int i = int(V::rank) - 1; i >= 0; i--)
        {
          if(expectedStride != smr.strides[i])
            Kokkos::abort("Cannot convert non-contiguous StridedMemRefType to LayoutRight Kokkos::View");
          expectedStride *= smr.sizes[i];
        }
      }
      return V(&smr.data[smr.offset], layout);
    }

  struct DualViewBase
  {
    virtual void syncHost();
    virtual void syncDevice();
    bool modified_host = false;
    bool modified_device = false;
  };

  template<typename DataType, typename Layout>
    struct DualView : public DualViewBase
  {
    static constexpr bool deviceAccessesHost = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;
    static constexpr bool hostAccessesDevice = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;

    using HostView = Kokkos::View<DataType, Layout, Kokkos::DefaultHostExecutionSpace>;
    using DeviceView = Kokkos::View<DataType, Layout, Kokkos::DefaultExecutionSpace>;

    // Constructor for allocating a new view.
    // Does not actually allocate anything yet; instead 
    DualView(
        const std::string& label,
        size_t ex0 = KOKKOS_INVALID_INDEX, size_t ex1 = KOKKOS_INVALID_INDEX, size_t ex2 = KOKKOS_INVALID_INDEX, size_t ex3 = KOKKOS_INVALID_INDEX,
        size_t ex4 = KOKKOS_INVALID_INDEX, size_t ex5 = KOKKOS_INVALID_INDEX, size_t ex6 = KOKKOS_INVALID_INDEX, size_t ex7 = KOKKOS_INVALID_INDEX)
    {
      if constexpr(hostAccessesDevice) {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_dev"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        host_view = HostView(device_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
      else if constexpr(deviceAccessesHost) {
        // Otherwise, host_view must be a separate allocation.
        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_host"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        device_view = DeviceView(host_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
      else {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_dev"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_host"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);
      }
      parent = this;
    }

    // Constructor which is given explicit device and host views, and a parent.
    // This can be used for subviewing/casting operations.
    // Note: d,h should view the same memory as parent, but they can
    // have a different data type and layout.
    DualView(DeviceView d, HostView h, DualViewBase* parent_)
      : device_view(d), host_view(h), parent(parent_)
    {}

    // Constructor for a host view from an external source (e.g. python)
    DualView(HostView h)
    {
      modified_host = true;
      if constexpr(deviceAccessesHost) {
        device_view = DeviceView(h.data(), h.layout());
      }
      else {
        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + "_dev"), h.layout());
      }
      host_view = h;
      parent = this;
    }

    void modifyHost()
    {
      parent->modified_host = true;
    }

    void modifyDevice()
    {
      parent->modified_device = true;
    }

    void syncHost() override
    {
      if constexpr(useSingleView) {
        Kokkos::fence();
      }
      else {
        if(parent == this) {
          Kokkos::deep_copy(host_view, device_view);
          modified_device = false;
        }
        else {
          parent->syncHost();
        }
      }
    }

    void syncDevice() override
    {
      // For single view, do not sync because
      // all host execution spaces are already synchronous
      if constexpr(!useSingleView) {
        if(parent == this) {
          Kokkos::deep_copy(device_view, host_view);
          modified_host = false;
        }
        else {
          parent->syncHost();
        }
      }
    }

    DeviceView device_view;
    HostView host_view;
    DualViewBase* parent;
  };
} // namespace LAPIS

extern "C" void lapis_initialize()
{
  if (!Kokkos::is_initialized()) Kokkos::initialize();
}

extern "C" void lapis_finalize()
{
  Kokkos::finalize();
}

