REPLICATE_USER ?= technillogue
model ?= dreambooth-batch

schema.json:
	cog run --use-cuda-base-image=false python3 -m cog.command.openapi_schema > schema.json
push: schema.json
	cog push --openapi-schema=schema.json --use-cuda-base-image=false --progress plain r8.im/$(REPLICATE_USER)/$(model)
