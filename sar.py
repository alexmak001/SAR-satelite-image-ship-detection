import ee 
import geemap

imgVV = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select('VV')

def func_bwd(image):
          edge = image.lt(-30.0)
          maskedImage = image.mask().And(edge.Not())
          return image.updateMask(maskedImage) \
        .map(func_bwd)






desc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
asc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

spring = ee.Filter.date('2015-03-01', '2015-04-20')
lateSpring = ee.Filter.date('2015-04-21', '2015-06-10')
summer = ee.Filter.date('2015-06-11', '2015-08-31')

descChange = ee.Image.cat(
        desc.filter(spring).mean(),
        desc.filter(lateSpring).mean(),
        desc.filter(summer).mean())

ascChange = ee.Image.cat(
        asc.filter(spring).mean(),
        asc.filter(lateSpring).mean(),
        asc.filter(summer).mean())

Map = geemap.Map()
Map.setCenter(5.2013, 47.3277, 12)
Map.addLayer(ascChange, {'min': -25, 'max': 5}, 'Multi-T Mean ASC', True)
Map.addLayer(descChange, {'min': -25, 'max': 5}, 'Multi-T Mean DESC', True)
Map