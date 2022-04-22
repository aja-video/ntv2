# aja apps
if (AJA_BUILD_APPS)
    # aja core demos
    aja_add_subdirectory(crossplatform/demoapps/ntv2burn)
    aja_add_subdirectory(crossplatform/demoapps/ntv2burn4kquadrant)
    aja_add_subdirectory(crossplatform/demoapps/ntv2capture)
    aja_add_subdirectory(crossplatform/demoapps/ntv2capture4k)
    aja_add_subdirectory(crossplatform/demoapps/ntv2capture8k)
    aja_add_subdirectory(crossplatform/demoapps/ntv2dolbyplayer)
    aja_add_subdirectory(crossplatform/demoapps/ntv2enumerateboards)
    aja_add_subdirectory(crossplatform/demoapps/ntv2fieldburn)
    aja_add_subdirectory(crossplatform/demoapps/ntv2hdrsetup)
    aja_add_subdirectory(crossplatform/demoapps/ntv2llburn)
    aja_add_subdirectory(crossplatform/demoapps/ntv2outputtestpattern)
    aja_add_subdirectory(crossplatform/demoapps/ntv2player)
    aja_add_subdirectory(crossplatform/demoapps/ntv2player4k)
    aja_add_subdirectory(crossplatform/demoapps/ntv2player8k)
    aja_add_subdirectory(crossplatform/demoapps/ntv2switchbitfile)

    # nvidia demos
    aja_add_subdirectory(crossplatform/demoapps/NVIDIA/cudalowlatencydemo)
    # aja_add_subdirectory(crossplatform/demoapps/NVIDIA/dvplowlatencydemo) # TODO(paulh): Need to audit dependencies for open-source distribution.

    # tools
    aja_add_subdirectory(crossplatform/logjammer)
    aja_add_subdirectory(crossplatform/logreader)
    aja_add_subdirectory(crossplatform/ntv2firmwareinstaller)
    aja_add_subdirectory(crossplatform/ntv2thermo)
    aja_add_subdirectory(crossplatform/pciwhacker)
    aja_add_subdirectory(crossplatform/rdmawhacker)
    aja_add_subdirectory(crossplatform/regio)
    aja_add_subdirectory(crossplatform/supportlog)
    aja_add_subdirectory(crossplatform/testaux)
    aja_add_subdirectory(crossplatform/testcrc)
    aja_add_subdirectory(crossplatform/testframe)
    aja_add_subdirectory(crossplatform/testp2p)
    aja_add_subdirectory(crossplatform/testrdma)
    aja_add_subdirectory(crossplatform/vpidtool)

    if (NOT AJA_BUILD_OPENSOURCE)
        # retail apps/services
        aja_add_subdirectory(crossplatform/ajadaemon "AJA Retail Daemon" "AJA Retail Daemon")
        aja_add_subdirectory(crossplatform/controlpanel "AJA ControlPanel" "AJA ControlPanel - Retail Device Configuration")
        aja_add_subdirectory(crossplatform/controlroom "AJA ControlRoom" "AJA Control Room - Ingest & Playout Application")
		if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND AJA_BUILD_NONAJA)
			# ndibridge is currently Linux-only, and requires building libsamplerate
			aja_add_subdirectory(crossplatform/ndibridge "AJA NDI Bridge" "AJA NDI Bridge - Make It So")
		endif()

        # oem/internal tools
        # NOTE(paulh): ajalogger is MIT licensed but need to remove dependency on ajagui/AJAAboutBox
        aja_add_subdirectory(crossplatform/ajalogger "AJA Logger" "AJA Logger - Log All The Things")
        aja_add_subdirectory(crossplatform/hdmiinstatus)
        aja_add_subdirectory(crossplatform/konaburner)
        aja_add_subdirectory(crossplatform/mcap)
        aja_add_subdirectory(crossplatform/ntv2cmd)
        aja_add_subdirectory(crossplatform/ntv2watcher "AJA NTV2 Watcher" "AJA NTV2 Watcher")
        # aja_add_subdirectory(crossplatform/rovocontrol "AJA RovoControl" "AJA RovoControl")
        aja_add_subdirectory(crossplatform/systemtest "AJA System Test" "AJA System Test")

        # aja cc demos
        aja_add_subdirectory(crossplatform/demoapps/ntv2ccgrabber)
        aja_add_subdirectory(crossplatform/demoapps/ntv2ccplayer)

        # aja cuda demos (TODO)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2cudacapture)
        # aja hevc demos
        # NOTE: These demos use proprietary code and are not included in the open-source NTV2 distribution.
        aja_add_subdirectory(crossplatform/demoapps/ntv2encodehevc)
        aja_add_subdirectory(crossplatform/demoapps/ntv2encodehevcfile)
        aja_add_subdirectory(crossplatform/demoapps/ntv2encodehevcvif)

        # aja opengl demos (TODO)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2oglcapture)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2ogldpxreader)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2testOGL)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2ogloutput)

        # aja Qt-based demos (TODO)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2dpxplayer)
        aja_add_subdirectory(crossplatform/demoapps/ntv2konaipj2ksetup)
        aja_add_subdirectory(crossplatform/demoapps/ntv2konaipjsonsetup)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2playcorder)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2recorddpx)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2stereocapture)
        # aja_add_subdirectory(crossplatform/demoapps/ntv2playbackdpx)
    endif()
endif()
