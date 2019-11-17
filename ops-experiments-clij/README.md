This repository contains a clij plugin for convolving images with custom kernels. Richardson-Lucy based deconvolution is available as well.

### Installation to ImageJ/Fiji

This plugin is delivered together with CLIJ via its update site. Add the update site http://sites.imagej.net/clij to your Fiji installation. [Read more about how to activate update sites]( https://imagej.net/Following_an_update_site)
After updating, restart Fiji.

### Example macro
You find some example macros in the folder [src/main/macro](src/main/macro).

### Development
If you want to develop this plugin, open pom.xml in your IDE. After you changed the code, to deploy a plugin to your Fiji installation, enter the correct path of your Fiji to the pom.xml file:

```xml
<imagej.app.directory>C:/programs/fiji-win64/Fiji.app/</imagej.app.directory>
```

Afterwards, run

```
mvn install
```

Restart Fiji and check using this macro if your plugin was installed successfully.
