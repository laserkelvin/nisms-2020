# geocoding.jl
module geocoding
include("HTTP")
include("JSON")

function getCoordinates(addressList)
    """
    getCoordinates(addressList::Array{String,1})

    returns a Vector of coordinates in (lat,lon) format as well as a data Array that contains all the data returned by nominatim
    """
    #nominatim requires as user agent and will block you otherwise
    HTTP.setuseragent!("Mozilla/5.0 (Windows NT 10.0; rv:68.0) Gecko/20100101 Firefox/68.0")
    coordinatesList = Array{Tuple{Float64,Float64},1}(undef, length(addressList)) # will contain (lat,lon) of addresses in addressList
    data = Array{Any,1}(undef, length(addressList)) # will contain all data nominatim gives for each request
    for (i, address) in enumerate(addressList)
        addressFormatted = join(split(address), "+")
        rawdata = HTTP.get(string(
            "https://nominatim.openstreetmap.org/search?q=",
            addressFormatted,
            "&format=json&limit=1",
        ))
        data[i] = JSON.parse(String(rawdata.body))[1]
        coordinatesList[i] = parse(Float64, data[i]["lat"]), parse(Float64, data[i]["lon"])
        sleep(1) #nominatim will block you if you have more than 1 request per second
    end
    return coordinatesList, data
end
