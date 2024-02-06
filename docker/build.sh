BUILD_ARGS_COMMON="
	--build-arg PLATFORM --rm -t torch_index/${PLATFORM}:${TAG} -f Dockerfile .
"

docker build ${BUILD_ARGS_COMMON}