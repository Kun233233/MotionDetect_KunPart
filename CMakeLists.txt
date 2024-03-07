cmake_minimum_required(VERSION 3.5.1)

#cmake ../samples/ -DBUILD_PLATFORM=aarch64/x86_64/armv7l	

project(SaveTest)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

MESSAGE(STATUS "BUILD_PLATFORM: ${BUILD_PLATFORM}") 
MESSAGE(STATUS "LIBRARY_SOURCE_DIR: ${LIBRARY_SOURCE_DIR}")
MESSAGE(STATUS "Project dir: ${PROJECT_SOURCE_DIR}")
MESSAGE(STATUS "Project binary: ${PROJECT_BINARY_DIR}")
MESSAGE(STATUS "cmake source dir: ${CMAKE_SOURCE_DIR}")
MESSAGE(STATUS "cmake system name: ${CMAKE_SYSTEM_NAME}")
MESSAGE(STATUS "cmake system processor: ${CMAKE_SYSTEM_PROCESSOR}")

# 应该是每次改这里就行
if((BUILD_PLATFORM MATCHES "aarch64") OR (BUILD_PLATFORM MATCHES "armv7l"))
MESSAGE(STATUS "add_subdirectory 1") 
	add_subdirectory(DepthSaveEvent)
	add_subdirectory(DepthSavePoll)
	add_subdirectory(ImagesRead)
	add_subdirectory(CameraCalibration)
	add_subdirectory(CalibrationImageCollection)
	add_subdirectory(DepthRGBRegistration)
	add_subdirectory(PCLTest)
	add_subdirectory(FeaturePointsInfo)
else()

	add_subdirectory(DepthSaveEvent)
	add_subdirectory(DepthSavePoll)
	add_subdirectory(ImagesRead)
	add_subdirectory(CameraCalibration)
	add_subdirectory(CalibrationImageCollection)
	add_subdirectory(DepthRGBRegistration)
	add_subdirectory(PCLTest)
	add_subdirectory(FeaturePointsInfo)

endif()




if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		MESSAGE(STATUS "Windows CMAKE_SIZEOF_VOID_P 8")
		set(LIBRARY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/OpenNI2/windows/x64-Release/)

	elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
		MESSAGE(STATUS "Windows CMAKE_SIZEOF_VOID_P 4")
		set(LIBRARY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/OpenNI2/windows/Win32-Release/)

	endif()

elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
	if(BUILD_PLATFORM MATCHES "x86_64")
                MESSAGE(STATUS "BUILD_PLATFORM 0: ${BUILD_PLATFORM}") 
		if(CMAKE_SIZEOF_VOID_P EQUAL 8)
			set(LIBRARY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/OpenNI2/linux/x64/)

		elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
			set(LIBRARY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/OpenNI2/linux/x86/)
		
		endif()

	elseif(BUILD_PLATFORM MATCHES "armv7l")
		MESSAGE(STATUS "BUILD_PLATFORM 1: ${BUILD_PLATFORM}")	
		set(LIBRARY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/OpenNI2/arm/Arm/)

    elseif(BUILD_PLATFORM MATCHES "aarch64")
		MESSAGE(STATUS "BUILD_PLATFORM 2: ${BUILD_PLATFORM}") 
		set(LIBRARY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/ThirdParty/OpenNI2/arm/Arm64/)
		
	endif()
endif()


MESSAGE(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "LIBRARY_SOURCE_DIR: ${LIBRARY_SOURCE_DIR}")
#该Copy方式比较局限，使用INSTALL做比较好。				
#add_custom_command(OUTPUT  COPY_RES  COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBRARY_SOURCE_DIR} ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}) 
#add_custom_target(CopyThirdPart ALL DEPENDS COPY_RES)


##Create INSTALL Project build
#openNI2
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	install(DIRECTORY ${LIBRARY_SOURCE_DIR} DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
	install(DIRECTORY ${LIBRARY_SOURCE_DIR} DESTINATION ${PROJECT_BINARY_DIR}/bin)
endif()


#Other
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	MESSAGE(STATUS "IS Windows platform")
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/GL/glut64.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/UvcSwapper/x64-Release/UVC-Swapper.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
		#opencv
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Windows/lib/win_x64/opencv_world430.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
		#D2C
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/d2c/lib/win64/release/d2c.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
	elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/GL/glut32.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/UvcSwapper/x86-Release/UVC-Swapper.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
		#opencv
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Windows/lib/win_x86/opencv_world430.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
		#D2C
		install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/d2c/lib/win32/release/d2c.dll DESTINATION ${PROJECT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
	endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
	MESSAGE(STATUS "IS Linux platform")
	if(BUILD_PLATFORM MATCHES "x86_64") #Linux
		if(CMAKE_SIZEOF_VOID_P EQUAL 8)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/UvcSwapper/x64/libUvc-Swapper.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			#opencv
			
			#libopencv_core
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_core.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_core.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_core.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_highgui
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_highgui.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_highgui.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_highgui.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_imgcodecs
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_imgcodecs.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_imgcodecs.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_imgcodecs.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_imgproc
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_imgproc.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_imgproc.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/Linux/lib/libopencv_imgproc.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			#D2C
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/d2c/lib/x64/libd2c.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			#libuvc
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/x64/libuvc.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/x64/libuvc.so.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/x64/libuvc.so.0.0.6 DESTINATION ${PROJECT_BINARY_DIR}/bin)
		elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)

		endif()

	elseif(BUILD_PLATFORM MATCHES "armv7l") #Arm v7
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/UvcSwapper/arm-32/libUvc-Swapper.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			#opencv
			
			#libopencv_core
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_core.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_core.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_core.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_highgui
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_highgui.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_highgui.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_highgui.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_imgcodecs
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_imgcodecs.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_imgcodecs.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_imgcodecs.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_imgproc
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_imgproc.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_imgproc.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_imgproc.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_videoio
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_videoio.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_videoio.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-32/lib/libopencv_videoio.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#D2C
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/d2c/lib/arm-32/libd2c.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			#libuvc
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/arm-32/libuvc.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/arm-32/libuvc.so.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/arm-32/libuvc.so.0.0.6 DESTINATION ${PROJECT_BINARY_DIR}/bin)
    elseif(BUILD_PLATFORM MATCHES "aarch64") #Arm v8
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/UvcSwapper/arm-64/libUvc-Swapper.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			#opencv
			#libopencv_core
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_core.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_core.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_core.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_highgui
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_highgui.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_highgui.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_highgui.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_imgcodecs
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_imgcodecs.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_imgcodecs.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_imgcodecs.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			#libopencv_imgproc
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_imgproc.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_imgproc.so.4.2 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/OpenCV420/arm-64/lib/libopencv_imgproc.so.4.2.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)

			
			#D2C
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/d2c/lib/arm-64/libd2c.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			#libuvc
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/arm-64/libuvc.so DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/arm-64/libuvc.so.0 DESTINATION ${PROJECT_BINARY_DIR}/bin)
			install(FILES ${PROJECT_SOURCE_DIR}/ThirdParty/libuvc/arm-64/libuvc.so.0.0.6 DESTINATION ${PROJECT_BINARY_DIR}/bin)

	endif()


endif()


