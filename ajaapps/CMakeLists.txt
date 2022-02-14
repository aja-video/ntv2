# aja apps
if (AJA_BUILD_APPS)
    # aja core demos
    add_subdirectory(crossplatform/demoapps/ntv2burn)
    add_subdirectory(crossplatform/demoapps/ntv2burn4kquadrant)
    add_subdirectory(crossplatform/demoapps/ntv2capture)
    add_subdirectory(crossplatform/demoapps/ntv2capture4k)
    add_subdirectory(crossplatform/demoapps/ntv2capture8k)
    add_subdirectory(crossplatform/demoapps/ntv2dolbyplayer)
    add_subdirectory(crossplatform/demoapps/ntv2enumerateboards)
    add_subdirectory(crossplatform/demoapps/ntv2fieldburn)
    add_subdirectory(crossplatform/demoapps/ntv2hdrsetup)
    add_subdirectory(crossplatform/demoapps/ntv2llburn)
    add_subdirectory(crossplatform/demoapps/ntv2outputtestpattern)
    add_subdirectory(crossplatform/demoapps/ntv2player)
    add_subdirectory(crossplatform/demoapps/ntv2player4k)
    add_subdirectory(crossplatform/demoapps/ntv2player8k)
    add_subdirectory(crossplatform/demoapps/ntv2switchbitfile)
    add_subdirectory(crossplatform/ntv2firmwareinstaller)

    # tools
    add_subdirectory(crossplatform/logjammer)
    add_subdirectory(crossplatform/logreader)
    add_subdirectory(crossplatform/ntv2thermo)
    add_subdirectory(crossplatform/pciwhacker)
    add_subdirectory(crossplatform/regio)
    add_subdirectory(crossplatform/testaux)
    add_subdirectory(crossplatform/testcrc)
    add_subdirectory(crossplatform/testframe)
    add_subdirectory(crossplatform/testp2p)
    add_subdirectory(crossplatform/vpidtool)

    if (NOT AJA_BUILD_OPENSOURCE)
        # retail apps/services
        add_subdirectory(crossplatform/ajadaemon)
        add_subdirectory(crossplatform/controlpanel)
        add_subdirectory(crossplatform/controlroom)
		if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
			# only on Linux for now
			add_subdirectory(crossplatform/ndibridge)
		endif()

        # oem/internal tools
        # NOTE(paulh): ajalogger is MIT licensed but need to remove dependency on ajagui/AJAAboutBox
        add_subdirectory(crossplatform/ajalogger)
        add_subdirectory(crossplatform/hdmiinstatus)
        add_subdirectory(crossplatform/konaburner)
        add_subdirectory(crossplatform/mcap)
        add_subdirectory(crossplatform/ntv2cmd)
        add_subdirectory(crossplatform/ntv2watcher)
        add_subdirectory(crossplatform/systemtest)

        # aja cc demos
        add_subdirectory(crossplatform/demoapps/ntv2ccgrabber)
        add_subdirectory(crossplatform/demoapps/ntv2ccplayer)

        # aja cuda demos (TODO)
        # add_subdirectory(crossplatform/demoapps/ntv2cudacapture)
        # aja hevc demos
        # NOTE: These demos use proprietary code and are not included in the open-source NTV2 distribution.
        add_subdirectory(crossplatform/demoapps/ntv2encodehevc)
        add_subdirectory(crossplatform/demoapps/ntv2encodehevcfile)
        add_subdirectory(crossplatform/demoapps/ntv2encodehevcvif)

        # aja opengl demos (TODO)
        # add_subdirectory(crossplatform/demoapps/ntv2oglcapture)
        # add_subdirectory(crossplatform/demoapps/ntv2ogldpxreader)
        # add_subdirectory(crossplatform/demoapps/ntv2testOGL)
        # add_subdirectory(crossplatform/demoapps/ntv2ogloutput)

        # aja Qt-based demos (TODO)
        # add_subdirectory(crossplatform/demoapps/ntv2dpxplayer)
        add_subdirectory(crossplatform/demoapps/ntv2konaipj2ksetup)
        add_subdirectory(crossplatform/demoapps/ntv2konaipjsonsetup)
        # add_subdirectory(crossplatform/demoapps/ntv2playcorder)
        # add_subdirectory(crossplatform/demoapps/ntv2recorddpx)
        # add_subdirectory(crossplatform/demoapps/ntv2stereocapture)
        # add_subdirectory(crossplatform/demoapps/ntv2playbackdpx)
    endif()
endif()
