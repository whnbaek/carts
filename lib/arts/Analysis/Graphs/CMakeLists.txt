# Graphs Analysis Implementation
# Source files for the graph components

# Main graph manager
file(GLOB CARTS_GRAPHS_SOURCE_FILES "*.cpp")

# Add Db graph subdirectory
add_subdirectory(Db)

# Add Edt graph subdirectory  
add_subdirectory(Edt)

# Add Concurrency graph subdirectory
add_subdirectory(Concurrency)

# Combine all graph source files
list(APPEND CARTS_GRAPHS_SOURCE_FILES ${CARTS_DB_GRAPHS_SOURCE_FILES})
list(APPEND CARTS_GRAPHS_SOURCE_FILES ${CARTS_EDT_GRAPHS_SOURCE_FILES})
list(APPEND CARTS_GRAPHS_SOURCE_FILES ${CONCURRENCY_GRAPHS_SOURCE_FILES})

# Set the source files for parent scope
set(CARTS_GRAPHS_SOURCE_FILES ${CARTS_GRAPHS_SOURCE_FILES} PARENT_SCOPE) 