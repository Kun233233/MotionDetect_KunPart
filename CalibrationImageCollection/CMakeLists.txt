set(TARGET CalibrationImageCollection) #这里要改
set(OPENCV_DLL_DIR "D:/aaaLab/aaagraduate/SaveVideo/ThirdParty/OpenCV470/opencv/build/x64/vc16/bin")

project(${TARGET})

include_directories(
    ../ThirdParty/Common
    ../ThirdParty/OpenNI2/Include
    # ../ThirdParty/OpenCV420/Windows/include
    #../ThirdParty/OpenCV470/opencv/build/include
	../ThirdParty/opencv-4.8.0/cudabuild/install/include/opencv2
	../ThirdParty/opencv-4.8.0/cudabuild/install/include
    ../ThirdParty/UvcSwapper/Include
)


if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	MESSAGE(STATUS "IS Windows platform")
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		MESSAGE(STATUS "IS Windows x64")
		# 64 bits
		link_directories(
			../ThirdParty/OpenNI2/windows/x64-Release
			# ../ThirdParty/OpenCV420/Windows/lib/win_x64
			# ../ThirdParty/OpenCV470/opencv/build/x64/vc16/lib
			# ../ThirdParty/OpenCV470/opencv/build/x64/vc16/bin
			../ThirdParty/opencv-4.8.0/cudabuild/install/x64/vc17/lib
			../ThirdParty/UvcSwapper/x64-Release
		)
	elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
		MESSAGE(STATUS "IS Windows x86") # 不要用X86
		# 32 bits
		link_directories(
			../ThirdParty/OpenNI2/windows/Win32-Release
			# ../ThirdParty/OpenCV420/Windows/lib/win_x86
			../ThirdParty/UvcSwapper/x86-Release
		)
	endif()

elseif(CMAKE_SYSTEM_NAME MATCHES "Linux") # linux的应该用不到，后面再改
	set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
	set(CMAKE_INSTALL_RPATH .)
	if(BUILD_PLATFORM MATCHES "x86_64")
		if(CMAKE_SIZEOF_VOID_P EQUAL 8)
			MESSAGE(STATUS "IS Linux x64")
			# 64 bits
			link_directories(
				../ThirdParty/OpenNI2/linux/x64
			)


		elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
			MESSAGE(STATUS "IS Linux x86")
			# 32 bits
			link_directories(
				../ThirdParty/OpenNI2/linux/x86
			)
		endif()

	elseif(BUILD_PLATFORM MATCHES "armv7l") # arm的应该用不到，后面再改
		MESSAGE(STATUS "IS armv7l")
		# 32 bits
		link_directories(
			../ThirdParty/OpenNI2/arm/Arm
		)
	elseif(BUILD_PLATFORM MATCHES "aarch64")
		MESSAGE(STATUS "IS armv8l")
		link_directories(
			../ThirdParty/OpenNI2/arm/Arm64
		)
	endif()
	
else()
	
endif()


#set(CMAKE_SKIP_RPATH FALSE)
#set(CMAKE_SKIP_BUILD_RPATH FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
#set(CMAKE_INSTALL_RPATH .)


#set(SRC_FILE DepthSavePoll.cpp)
file(GLOB SRC "*.h" "*.cpp")

#add_executable(${TARGET} ${SRC_FILE})
add_executable(${TARGET} ${SRC})

target_link_libraries(${TARGET} OpenNI2 Uvc-Swapper opencv_world480)

#set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH "${ORIGIN}" )


# 将 OpenCV 的 DLL 文件复制到程序输出目录中
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_DLL_DIR}/opencv_world480.dll ${CMAKE_BINARY_DIR}/Release
# 	COMMAND ${CMAKE_COMMAND} -E copy ${OPENCV_DLL_DIR}/opencv_world480.pdb ${CMAKE_BINARY_DIR}/Release
# )





