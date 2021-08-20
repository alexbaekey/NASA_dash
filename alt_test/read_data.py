import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
plt.ion()
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd


#38 -78 bottom left
#39.75 -74 top right
#obtained from https://topex.ucsd.edu/cgi-bin/get_data.cgi
data=np.loadtxt('data/baltimore_altitude2.txt')

states = gpd.read_file('data/usa-states-census-2014.shp')

# state boundary shape file obtained from 
#https://www.arcgis.com/home/item.html?id=f7f805eb65eb4ab787a0a3e1116ca7e5
#https://medium.com/@erikgreenj/mapping-us-states-with-geopandas-made-simple-d7b6e66fa20d
Long = data[:,0] 
Lat  = data[:,1] 
Elev = data[:,2]

pts = 100000

t1 = np.linspace(np.min(Long),np.max(Long),int(np.sqrt(pts)))
t2 = np.linspace(np.min(Lat),np.max(Lat),int(np.sqrt(pts)))

[x,y]=np.meshgrid(t1,t2)

z=griddata((Long,Lat),Elev, (x,y), method='linear')
x=np.matrix.flatten(x)
y=np.matrix.flatten(y)
z=np.matrix.flatten(z)

plt.scatter(x,y,1,z,cmap='terrain')
plt.colorbar(label='elevation above sea level (m)')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
ax = plt.gca()
ax.set_aspect('equal')
ax.add_patch(states[states['NAME'] == 'Maryland'])

#states_provinces = cfeature.NaturalEarthFeature(
#    category='cultural',
#    name = 'admin_1_states_provinces_lines',
#    scale='50m',
#    facecolor='none')

usa = gpd.read_file('data/states_21basic/states.shp')

x=np.matrix.flatten(x)




