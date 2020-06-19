using CSV, DataFrames, Plots, StatsPlots, Missings
using MLDataUtils
using PyCall
using JSON, HTTP
using Dates

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
        # exception if we aren't able to parse lat/long for any reason
        # we simply set the values to zero
        try
            data[i] = JSON.parse(String(rawdata.body))[1]
            coordinatesList[i] = parse(Float64, data[i]["lat"]), parse(Float64, data[i]["lon"])
            sleep(1) #nominatim will block you if you have more than 1 request per second
        catch e
            coordinatesList[i] = (0., 0.)
        end
    end
    return coordinatesList, data
end

P = download("https://docs.google.com/spreadsheets/d/1uv0IOtBCdpql3y926MtoT0wrHU3T7KGtLKIJ1FZMkjo/export?format=csv&gid=176397632", "responses.csv")

data = CSV.read("responses.csv");
# remove the last two columns where Marie wrote some stuff
data = data[:,1:7];

# rename the columns
rename!(data, [:Date, :Email, :Name, :TimeZone, :Org, :Position, :Hotdog]);

# remove entries with missing responses
dropmissing!(data);

# encode hotdog responses as bool for logistic regression later
hotdog_bool = Dict("Yes" => Float32(1.), "No" => Float32(0.));

# Create a new column
data.HotDogEncoding = map(val -> hotdog_bool[val], data.Hotdog);

# Generate encodings for each postiion type
data[!, :PosEncoding] = convertlabel(LabelEnc.Indices{Float32}, data.Position);

# Instead of running it for every row in the dataframe, do it on only
# the unique organizations to cut down time in half!
orgs = unique(data[:,:Org]);
coordinates = getCoordinates(orgs);
# Generate a mapping dictionary that will fill in blanks
coord_mapping = Dict(org => coord for (org, coord) in zip(orgs, coordinates[1]))
coord_mapping["Center for Astrophysics | Harvard & Smithsonian"] = (42.381405, -71.128036);
coord_mapping["Fritz-Haber Institute"] = (52.448343, 13.282765);
for key in ["U Paris Saclay/CNRS/ISMO", "CNRS/Université Paris Saclay", "University of Paris-Saclay"]
    coord_mapping[key] = (48.706617, 2.178913);
end
coord_mapping["Institut de Physique de Rennes"] = (48.116743, -1.640843);
coord_mapping["University of Reims Champagne-Ardenne"] = (49.255836, 4.040983);
coord_mapping["New England Biolab"] = (42.650274, -70.842718);
coord_mapping["Laboratoire de Physique des Lasers, CNRS-Université Sorbone Paris Nord"] = (48.956389, 2.340612);
coord_mapping["Leiden Observatory"] = (52.154681, 4.483809);
coord_mapping["Imperial College London (Earth Science & Engineering)"] = (51.498866, -0.175457);
coord_mapping["JILA, University of Colorado Boulder"] = coord_mapping["University of Colorado Boulder"];
coord_mapping["Laboratoire Interdiscipliniare de Physique (LIPhy)"] = coord_mapping["Laboratoire interdisciplinaire de Physique"];
coord_mapping["LMD  Ecole Molytechnique, France"] = (48.714226, 2.210970);
# Get lat, long for all the institutions
# coords = getCoordinates(data[:,:Org]);

data[!,:temp] = map(loc->coord_mapping[loc], data.Org);
lat_array = [];
long_array = [];
for coord in data.temp
    lat, long = coord
    push!(lat_array, lat)
    push!(long_array, long)
end
data[!,:Lat] = lat_array;
data[!,:Long] = long_array;

# Generate a relative time from first registration
reg_times = DateTime.(data.Date, dateformat"d/m/y H:M:S");
reg_times .-= reg_times[1]
relative_time = Dates.value.(reg_times) ./ (1000. * 60. * 60. * 24.);
data[!,:RelativeTime] = relative_time;

CSV.write("cleaned.csv", data);
