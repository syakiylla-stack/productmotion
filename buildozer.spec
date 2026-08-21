[app]

title = Product Motion V5.2
package.name = productmotion
package.domain = org.productmotion

source.dir = .
source.include_exts = py,png,jpg,jpeg,webp,kv,atlas

version = 5.2

requirements = python3,kivy,numpy,opencv,plyer,pyjnius

orientation = portrait

fullscreen = 0

android.api = 35
android.minapi = 24
android.archs = arm64-v8a

android.permissions = READ_MEDIA_IMAGES,READ_MEDIA_VIDEO,WRITE_EXTERNAL_STORAGE

[buildozer]

log_level = 2
warn_on_root = 1
