add_subdirectory(run)

# Install the main carts wrapper script into the same bin as other tools
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/carts
  DESTINATION ${CARTS_TOOLS_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  COMPONENT carts
)