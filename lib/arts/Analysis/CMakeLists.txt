# Add the main analysis source files
file(GLOB CARTS_ANALYSIS_SOURCE_FILES "*.cpp")

# Add the Db subdirectory
add_subdirectory(Db)

# Add the EDT subdirectory
add_subdirectory(Edt)

# Add the Concurrency subdirectory
add_subdirectory(Concurrency)

# Add the Graphs subdirectory
add_subdirectory(Graphs)

# Combine all analysis source files
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CARTS_DB_ANALYSIS_SOURCE_FILES})
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CARTS_EDT_ANALYSIS_SOURCE_FILES})
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CONCURRENCY_SOURCE_FILES})
list(APPEND CARTS_ANALYSIS_SOURCE_FILES ${CARTS_GRAPHS_SOURCE_FILES})

set(CARTS_ANALYSIS_SOURCE_FILES ${CARTS_ANALYSIS_SOURCE_FILES} PARENT_SCOPE)