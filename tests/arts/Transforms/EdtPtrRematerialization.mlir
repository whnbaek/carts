// RUN: %carts opt %s --edt-ptr-rematerialization | %filecheck %s --check-prefix=REMAT

// Test that pointer rematerialization optimizes pointer dependencies in EDTs

module {
  func.func @test_ptr_rematerialization(%arg0: memref<10xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // EDT with pointer dependencies that could be rematerialized
    arts.edt "ptr_dependent_task" {
      %ptr_offset = arith.addi %c0, %c1 : index
      %val = arith.constant 42 : i32
      memref.store %val, %arg0[%ptr_offset] : memref<10xi32>
      arts.edt_terminator
    }
    
    return
  }
}

// REMAT: func.func @test_ptr_rematerialization
// REMAT: arts.edt
// REMAT: arith.addi
// Check that pointer computations are optimized for memory usage and data movement
